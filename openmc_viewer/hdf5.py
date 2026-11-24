"""
A small, read-only HDF5 reader written in pure Python (standard library only).

It understands the parts of the HDF5 format that OpenMC (and h5py with its
default settings) produce:

* superblock versions 0, 1, 2 and 3
* object header versions 1 and 2
* old-style groups (symbol tables) and new-style groups (compact and dense
  link storage, fractal heaps, version-2 B-trees)
* attributes (compact and dense storage)
* contiguous, compact and chunked datasets, with the deflate (gzip),
  shuffle and fletcher32 filters
* integer, float, string (fixed and variable length), compound, enum,
  array, reference, bitfield and opaque datatypes

Usage::

    f = File("statepoint.100.h5")
    for name, node in f.root.items():
        print(name, node)
    ds = f["tallies/tally 1/results"]
    print(ds.shape, ds.dtype, ds.read()[:10])
"""

import itertools
import math
import mmap
import os
import struct
import sys
import zlib
from array import array

__all__ = ["File", "Group", "Dataset", "Datatype", "HDF5Error", "is_hdf5"]

SIGNATURE = b"\x89HDF\r\n\x1a\n"
UNDEFINED = object()


class HDF5Error(Exception):
    """Raised when a file cannot be read."""


def is_hdf5(path):
    try:
        with open(path, "rb") as fh:
            for off in (0, 512, 1024, 2048, 4096):
                fh.seek(off)
                if fh.read(8) == SIGNATURE:
                    return True
    except OSError:
        pass
    return False


def _pad8(n):
    return (n + 7) & ~7


def _prod(seq):
    r = 1
    for x in seq:
        r *= x
    return r


def _cstring(buf, pos):
    end = buf.find(b"\x00", pos)
    if end < 0:
        end = len(buf)
    return bytes(buf[pos:end]), end - pos


def _decode_text(raw):
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("latin-1")


# --------------------------------------------------------------------------
# Datatypes
# --------------------------------------------------------------------------

class Datatype(object):
    INT, FLOAT, TIME, STRING, BITFIELD, OPAQUE, COMPOUND, REFERENCE, ENUM, \
        VLEN, ARRAY = range(11)

    def __init__(self, cls, size):
        self.cls = cls
        self.size = size
        self.order = "<"
        self.signed = False
        self.members = []      # compound: (name, offset, Datatype)
        self.base = None       # enum / vlen / array
        self.enum = {}         # value -> name
        self.dims = ()         # array
        self.strpad = 0
        self.vlen_string = False

    # Human readable name, e.g. "float64", "string[20]"
    def __str__(self):
        c = self.cls
        if c == self.INT or c == self.TIME:
            return "%sint%d" % ("" if self.signed else "u", self.size * 8)
        if c == self.FLOAT:
            return "float%d" % (self.size * 8)
        if c == self.STRING:
            return "string[%d]" % self.size
        if c == self.BITFIELD:
            return "bitfield%d" % (self.size * 8)
        if c == self.OPAQUE:
            return "opaque[%d]" % self.size
        if c == self.REFERENCE:
            return "reference"
        if c == self.COMPOUND:
            return "compound{%s}" % ", ".join(
                "%s: %s" % (m[0], m[2]) for m in self.members)
        if c == self.ENUM:
            if sorted(self.enum.values()) == ["FALSE", "TRUE"]:
                return "bool"
            return "enum(%s)" % ", ".join(self.enum.values())
        if c == self.VLEN:
            return "string (variable length)" if self.vlen_string else \
                "sequence of %s" % self.base
        if c == self.ARRAY:
            return "%s%s" % (self.base, list(self.dims))
        return "unknown"

    @property
    def is_numeric(self):
        return self.cls in (self.INT, self.FLOAT, self.TIME, self.BITFIELD)

    @property
    def is_string(self):
        return self.cls == self.STRING or (self.cls == self.VLEN and self.vlen_string)


_INT_CODES = {(1, True): "b", (1, False): "B", (2, True): "h", (2, False): "H",
              (4, True): "i", (4, False): "I", (8, True): "q", (8, False): "Q"}
_FLOAT_CODES = {4: "f", 8: "d"}


# --------------------------------------------------------------------------
# The file
# --------------------------------------------------------------------------

class File(object):
    """An HDF5 file opened for reading."""

    def __init__(self, path):
        self.path = path
        self.filename = os.path.basename(path)
        self._fh = open(path, "rb")
        size = os.fstat(self._fh.fileno()).st_size
        if size == 0:
            raise HDF5Error("The file is empty.")
        self.buf = mmap.mmap(self._fh.fileno(), 0, access=mmap.ACCESS_READ)
        self._gheap_cache = {}
        self._fheap_cache = {}
        self._header_cache = {}
        self._read_superblock()
        self.root = Group(self, "/", "/", self.root_addr)

    # -- context manager ----------------------------------------------------
    def close(self):
        try:
            self.buf.close()
        except Exception:
            pass
        self._fh.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __getitem__(self, path):
        return self.root[path]

    # -- low-level helpers ---------------------------------------------------
    def u8(self, p):
        return self.buf[p]

    def u16(self, p):
        return struct.unpack_from("<H", self.buf, p)[0]

    def u32(self, p):
        return struct.unpack_from("<I", self.buf, p)[0]

    def u64(self, p):
        return struct.unpack_from("<Q", self.buf, p)[0]

    def uint(self, p, n):
        return int.from_bytes(self.buf[p:p + n], "little")

    def addr(self, p):
        """Read an address; returns None for the 'undefined address'."""
        v = self.uint(p, self.O)
        if v == (1 << (8 * self.O)) - 1:
            return None
        return v + self.base

    def length(self, p):
        return self.uint(p, self.L)

    def _check_sig(self, p, sig):
        if self.buf[p:p + len(sig)] != sig:
            raise HDF5Error("Corrupt or unsupported file: expected %r at %d"
                            % (sig, p))

    # -- superblock ------------------------------------------------------
    def _read_superblock(self):
        sb = None
        off = 0
        while off < len(self.buf):
            if self.buf[off:off + 8] == SIGNATURE:
                sb = off
                break
            off = 512 if off == 0 else off * 2
        if sb is None:
            raise HDF5Error("This is not an HDF5 file.")
        ver = self.u8(sb + 8)
        self.superblock_version = ver
        if ver in (0, 1):
            self.O = self.u8(sb + 13)
            self.L = self.u8(sb + 14)
            p = sb + 24
            if ver == 1:
                p += 4
            self.base = 0
            base = self.uint(p, self.O)
            p += 4 * self.O                  # base, free space, EOF, driver
            self.base = base
            # root group symbol table entry
            self.root_addr = self.addr(p + self.O)
        elif ver in (2, 3):
            self.O = self.u8(sb + 9)
            self.L = self.u8(sb + 10)
            p = sb + 12
            self.base = 0
            self.base = self.uint(p, self.O)
            p += 3 * self.O                  # base, extension, EOF
            self.root_addr = self.addr(p)
        else:
            raise HDF5Error("Unsupported HDF5 superblock version %d" % ver)

    # -- object headers --------------------------------------------------
    def messages(self, addr):
        """Return the list of (type, flags, data_pos, size) of an object."""
        if addr in self._header_cache:
            return self._header_cache[addr]
        msgs = []
        buf = self.buf
        if buf[addr:addr + 4] == b"OHDR":
            flags = buf[addr + 5]
            p = addr + 6
            if flags & 0x20:
                p += 16
            if flags & 0x10:
                p += 4
            n = 1 << (flags & 3)
            size0 = self.uint(p, n)
            p += n
            hdr = 6 if flags & 0x04 else 4
            blocks = [(p, size0)]
            while blocks:
                start, size = blocks.pop(0)
                q, end = start, start + size
                while q + hdr <= end:
                    mtype = buf[q]
                    msize = struct.unpack_from("<H", buf, q + 1)[0]
                    mflags = buf[q + 3]
                    data = q + hdr
                    if mtype == 0x10:
                        caddr = self.addr(data)
                        clen = self.length(data + self.O)
                        self._check_sig(caddr, b"OCHK")
                        blocks.append((caddr + 4, clen - 8))
                    elif mtype != 0:
                        msgs.append((mtype, mflags, data, msize))
                    q = data + msize
        elif buf[addr] == 1:
            hsize = self.u32(addr + 8)
            blocks = [(addr + 16, hsize)]
            while blocks:
                start, size = blocks.pop(0)
                q, end = start, start + size
                while q + 8 <= end:
                    mtype, msize, mflags = struct.unpack_from("<HHB", buf, q)
                    data = q + 8
                    if mtype == 0x10:
                        blocks.append((self.addr(data),
                                       self.length(data + self.O)))
                    elif mtype != 0:
                        msgs.append((mtype, mflags, data, msize))
                    q = data + msize
        else:
            raise HDF5Error("Unsupported object header at address %d" % addr)
        self._header_cache[addr] = msgs
        return msgs

    def node_at(self, addr, name, path):
        msgs = self.messages(addr)
        types = set(m[0] for m in msgs)
        if 0x0008 in types and 0x0003 in types:
            return Dataset(self, name, path, addr)
        if 0x0003 in types and 0x0001 not in types and not (types & {0x02, 0x06, 0x11}):
            return CommittedDatatype(self, name, path, addr)
        return Group(self, name, path, addr)

    # -- datatypes ---------------------------------------------------------
    def parse_datatype(self, p):
        """Parse a datatype message at p; returns (Datatype, bytes_used)."""
        buf = self.buf
        b0 = buf[p]
        cls = b0 & 0x0F
        ver = b0 >> 4
        bits0, bits1 = buf[p + 1], buf[p + 2]
        size = self.u32(p + 4)
        dt = Datatype(cls, size)
        if cls in (Datatype.INT, Datatype.BITFIELD):
            dt.order = ">" if bits0 & 1 else "<"
            dt.signed = bool(bits0 & 0x08) and cls == Datatype.INT
            return dt, 12
        if cls == Datatype.FLOAT:
            dt.order = ">" if bits0 & 1 else "<"
            return dt, 20
        if cls == Datatype.TIME:
            dt.order = ">" if bits0 & 1 else "<"
            dt.signed = True
            return dt, 10
        if cls == Datatype.STRING:
            dt.strpad = bits0 & 0x0F
            return dt, 8
        if cls == Datatype.OPAQUE:
            return dt, 8 + bits0
        if cls == Datatype.REFERENCE:
            return dt, 8
        if cls == Datatype.COMPOUND:
            nmemb = bits0 | (bits1 << 8)
            q = p + 8
            for _ in range(nmemb):
                raw, n = _cstring(buf, q)
                name = _decode_text(raw)
                if ver < 3:
                    q += _pad8(n + 1)
                    moff = self.u32(q)
                    q += 4
                    adims = []
                    if ver == 1:
                        nd = buf[q]
                        adims = [self.u32(q + 12 + 4 * i) for i in range(nd)]
                        q += 28
                    mt, used = self.parse_datatype(q)
                    q += used
                    if adims:
                        at = Datatype(Datatype.ARRAY, mt.size * _prod(adims))
                        at.base, at.dims = mt, tuple(adims)
                        mt = at
                else:
                    q += n + 1
                    nb = 1 if size < 256 else 2 if size < 65536 else \
                        3 if size < (1 << 24) else 4
                    moff = self.uint(q, nb)
                    q += nb
                    mt, used = self.parse_datatype(q)
                    q += used
                dt.members.append((name, moff, mt))
            return dt, q - p
        if cls == Datatype.ENUM:
            nmemb = bits0 | (bits1 << 8)
            q = p + 8
            base, used = self.parse_datatype(q)
            q += used
            dt.base = base
            names = []
            for _ in range(nmemb):
                raw, n = _cstring(buf, q)
                names.append(_decode_text(raw))
                q += _pad8(n + 1) if ver < 3 else n + 1
            values = self._decode(base, bytes(buf[q:q + nmemb * base.size]), nmemb)
            q += nmemb * base.size
            dt.enum = dict(zip(values, names))
            return dt, q - p
        if cls == Datatype.VLEN:
            dt.vlen_string = (bits0 & 0x0F) == 1
            base, used = self.parse_datatype(p + 8)
            dt.base = base
            return dt, 8 + used
        if cls == Datatype.ARRAY:
            q = p + 8
            nd = buf[q]
            q += 4 if ver == 2 else 1
            dt.dims = tuple(self.u32(q + 4 * i) for i in range(nd))
            q += 4 * nd
            if ver == 2:
                q += 4 * nd
            base, used = self.parse_datatype(q)
            dt.base = base
            return dt, q + used - p
        raise HDF5Error("Unsupported datatype class %d" % cls)

    def shared_datatype(self, p):
        """Resolve a shared (committed) datatype message."""
        ver = self.u8(p)
        typ = self.u8(p + 1)
        if ver == 1:
            a = self.addr(p + 8)
        elif ver == 2 or (ver == 3 and typ == 2):
            a = self.addr(p + 2)
        else:
            raise HDF5Error("Shared messages in the shared-message heap are "
                            "not supported")
        for mtype, mflags, data, msize in self.messages(a):
            if mtype == 0x0003:
                return self.parse_datatype(data)[0]
        raise HDF5Error("Committed datatype not found")

    def parse_dataspace(self, p):
        """Return the shape tuple; () for scalar, None for null dataspace."""
        ver = self.u8(p)
        nd = self.u8(p + 1)
        if ver == 1:
            q = p + 8
        else:
            q = p + 4
            if self.u8(p + 3) == 2:
                return None
        return tuple(self.length(q + i * self.L) for i in range(nd))

    # -- global heap (variable length data) --------------------------------
    def global_heap_object(self, addr, index):
        coll = self._gheap_cache.get(addr)
        if coll is None:
            coll = {}
            self._check_sig(addr, b"GCOL")
            size = self.length(addr + 8)
            p = addr + 8 + self.L
            end = addr + size
            while p + 8 + self.L <= end:
                idx = self.u16(p)
                if idx == 0:
                    break
                osize = self.length(p + 8)
                d = p + 8 + self.L
                coll[idx] = (d, osize)
                p = d + _pad8(osize)
            self._gheap_cache[addr] = coll
        d, osize = coll[index]
        return bytes(self.buf[d:d + osize])

    # -- decoding raw bytes into python values ---------------------------
    def _decode(self, dt, raw, count):
        """Decode `count` consecutive elements of type `dt` from raw bytes."""
        c = dt.cls
        size = dt.size
        if count == 0:
            return []
        if c in (Datatype.INT, Datatype.TIME, Datatype.BITFIELD) or \
                (c == Datatype.FLOAT and size in (4, 8)):
            if c == Datatype.FLOAT:
                code = _FLOAT_CODES[size]
            else:
                code = _INT_CODES.get((size, dt.signed))
            if code is not None and array(code).itemsize == size:
                a = array(code)
                a.frombytes(raw[:count * size])
                if (dt.order == ">") != (sys.byteorder == "big"):
                    a.byteswap()
                return a.tolist()
            signed = dt.signed
            order = "big" if dt.order == ">" else "little"
            return [int.from_bytes(raw[i * size:(i + 1) * size], order, signed=signed)
                    for i in range(count)]
        if c == Datatype.FLOAT:
            if size == 2:
                fmt = dt.order + "%de" % count
                return list(struct.unpack(fmt, raw[:2 * count]))
            raise HDF5Error("Unsupported float size %d" % size)
        if c == Datatype.STRING:
            out = []
            for i in range(count):
                s = raw[i * size:(i + 1) * size]
                if dt.strpad == 2:          # space padded
                    s = s.rstrip(b" ")
                else:
                    z = s.find(b"\x00")
                    if z >= 0:
                        s = s[:z]
                out.append(_decode_text(s))
            return out
        if c == Datatype.VLEN:
            out = []
            step = 4 + self.O + 4
            for i in range(count):
                q = i * step
                n = struct.unpack_from("<I", raw, q)[0]
                gaddr = int.from_bytes(raw[q + 4:q + 4 + self.O], "little")
                idx = struct.unpack_from("<I", raw, q + 4 + self.O)[0]
                if gaddr == 0 or n == 0 and idx == 0:
                    out.append("" if dt.vlen_string else [])
                    continue
                data = self.global_heap_object(gaddr + self.base, idx)
                if dt.vlen_string:
                    out.append(_decode_text(data.rstrip(b"\x00")))
                else:
                    out.append(self._decode(dt.base, data, n))
            return out
        if c == Datatype.ENUM:
            vals = self._decode(dt.base, raw, count)
            if sorted(dt.enum.values()) == ["FALSE", "TRUE"]:
                return [dt.enum.get(v) == "TRUE" for v in vals]
            return [dt.enum.get(v, v) for v in vals]
        if c == Datatype.COMPOUND:
            cols = []
            for name, off, mt in dt.members:
                mraw = b"".join(raw[i * size + off:i * size + off + mt.size]
                                for i in range(count))
                cols.append(self._decode(mt, mraw, count))
            names = [m[0] for m in dt.members]
            return [dict(zip(names, vals)) for vals in zip(*cols)]
        if c == Datatype.ARRAY:
            n = _prod(dt.dims)
            flat = self._decode(dt.base, raw, count * n)
            return [_reshape(flat[i * n:(i + 1) * n], dt.dims) for i in range(count)]
        if c == Datatype.REFERENCE:
            return ["<object @%d>" % int.from_bytes(raw[i * size:(i + 1) * size], "little")
                    for i in range(count)]
        if c == Datatype.OPAQUE:
            return [raw[i * size:(i + 1) * size].hex() for i in range(count)]
        raise HDF5Error("Cannot decode datatype %s" % dt)

    # -- fractal heaps and v2 B-trees (new style groups) -----------------
    def fractal_heap(self, addr):
        h = self._fheap_cache.get(addr)
        if h is None:
            h = _FractalHeap(self, addr)
            self._fheap_cache[addr] = h
        return h

    def btree2_records(self, addr):
        f = self
        f._check_sig(addr, b"BTHD")
        node_size = f.u32(addr + 6)
        rsize = f.u16(addr + 10)
        depth = f.u16(addr + 12)
        root = f.addr(addr + 16)
        root_nrec = f.u16(addr + 16 + f.O)
        if root is None:
            return []

        def enc_size(n):
            return (max(n, 1).bit_length() - 1) // 8 + 1

        max_nrec = [(node_size - 10) // rsize]
        cum = [max_nrec[0]]
        cum_size = [0]
        max_nrec_size = enc_size(max_nrec[0])
        for u in range(1, depth + 1):
            ptr = f.O + max_nrec_size + (cum_size[u - 1] if u > 1 else 0)
            m = (node_size - (10 + ptr)) // (rsize + ptr)
            max_nrec.append(m)
            cum.append((m + 1) * cum[u - 1] + m)
            cum_size.append(enc_size(cum[u]))

        out = []

        def walk(node, nrec, d):
            p = node + 6
            if d == 0:
                f._check_sig(node, b"BTLF")
                for i in range(nrec):
                    out.append(bytes(f.buf[p + i * rsize:p + (i + 1) * rsize]))
                return
            f._check_sig(node, b"BTIN")
            q = p + nrec * rsize
            for i in range(nrec + 1):
                child = f.addr(q)
                q += f.O
                cn = f.uint(q, max_nrec_size)
                q += max_nrec_size
                if d > 1:
                    q += cum_size[d - 1]
                walk(child, cn, d - 1)
                if i < nrec:
                    out.append(bytes(f.buf[p + i * rsize:p + (i + 1) * rsize]))

        walk(root, root_nrec, depth)
        return out

    # -- links of a group --------------------------------------------------
    def parse_link(self, p):
        """Parse a link message; returns (name, kind, target)."""
        buf = self.buf
        flags = buf[p + 1]
        q = p + 2
        ltype = 0
        if flags & 0x08:
            ltype = buf[q]
            q += 1
        if flags & 0x04:
            q += 8
        if flags & 0x10:
            q += 1
        n = 1 << (flags & 3)
        nlen = self.uint(q, n)
        q += n
        name = _decode_text(bytes(buf[q:q + nlen]))
        q += nlen
        if ltype == 0:
            return name, "hard", self.addr(q)
        if ltype == 1:
            ln = self.u16(q)
            return name, "soft", _decode_text(bytes(buf[q + 2:q + 2 + ln]))
        if ltype == 64:
            ln = self.u16(q)
            parts = bytes(buf[q + 3:q + 2 + ln]).split(b"\x00")
            return name, "external", "%s:%s" % (
                _decode_text(parts[0]), _decode_text(parts[1]) if len(parts) > 1 else "")
        return name, "unknown", None

    def symbol_table_links(self, btree, heap):
        self._check_sig(heap, b"HEAP")
        heap_data = self.addr(heap + 8 + 2 * self.L)
        links = []

        def walk(node):
            self._check_sig(node, b"TREE")
            level = self.u8(node + 5)
            n = self.u16(node + 6)
            p = node + 8 + 2 * self.O
            for i in range(n):
                child = self.addr(p + self.L)
                p += self.L + self.O
                if level > 0:
                    walk(child)
                else:
                    self._check_sig(child, b"SNOD")
                    ns = self.u16(child + 6)
                    e = child + 8
                    for _ in range(ns):
                        noff = self.length(e)
                        oaddr = self.addr(e + self.O)
                        ctype = self.u32(e + 2 * self.O)
                        raw, _n = _cstring(self.buf, heap_data + noff)
                        name = _decode_text(raw)
                        if ctype == 2:   # soft link stored in scratch pad
                            loff = self.u32(e + 2 * self.O + 8)
                            raw2, _n2 = _cstring(self.buf, heap_data + loff)
                            links.append((name, "soft", _decode_text(raw2)))
                        else:
                            links.append((name, "hard", oaddr))
                        e += 2 * self.O + 24
        walk(btree)
        return links


class _FractalHeap(object):
    def __init__(self, f, addr):
        self.f = f
        f._check_sig(addr, b"FRHP")
        p = addr + 5
        self.id_len = f.u16(p)
        filt_len = f.u16(p + 2)
        flags = f.u8(p + 4)
        max_managed = f.u32(p + 5)
        p += 9
        p += f.L + f.O + f.L + f.O          # next huge id, huge btree, free, fsm
        p += 8 * f.L                         # space/object counters
        self.width = f.u16(p)
        p += 2
        self.start = f.length(p)
        p += f.L
        max_direct = f.length(p)
        p += f.L
        max_heap_bits = f.u16(p)
        p += 2
        p += 2                               # starting rows
        root = f.addr(p)
        p += f.O
        cur_rows = f.u16(p)
        self.filtered = filt_len > 0
        self.off_size = (max_heap_bits + 7) // 8
        self.len_size = min((int(math.log2(max_direct)) + 7) // 8,
                            (max(max_managed, 1).bit_length() - 1) // 8 + 1)
        self.max_direct_rows = int(math.log2(max_direct)) - int(math.log2(self.start)) + 2
        self.blocks = []
        if root is not None:
            if cur_rows == 0:
                self._add_direct(root, self.start)
            else:
                self._walk_indirect(root, cur_rows)
        self.blocks.sort()

    def _row_size(self, r):
        return self.start if r < 2 else self.start * (1 << (r - 1))

    def _add_direct(self, addr, size):
        f = self.f
        f._check_sig(addr, b"FHDB")
        boff = f.uint(addr + 5 + f.O, self.off_size)
        self.blocks.append((boff, addr, size))

    def _walk_indirect(self, addr, nrows):
        f = self.f
        f._check_sig(addr, b"FHIB")
        p = addr + 5 + f.O + self.off_size
        for r in range(nrows):
            for _c in range(self.width):
                child = f.addr(p)
                p += f.O
                if r < self.max_direct_rows:
                    if self.filtered:
                        p += f.L + 4
                    if child is not None:
                        self._add_direct(child, self._row_size(r))
                elif child is not None:
                    rows = int(math.log2(self._row_size(r))) - \
                        int(math.log2(self.start * self.width)) + 1
                    self._walk_indirect(child, rows)

    def get(self, hid):
        kind = (hid[0] >> 4) & 0x03
        if kind == 0:
            off = int.from_bytes(hid[1:1 + self.off_size], "little")
            ln = int.from_bytes(hid[1 + self.off_size:1 + self.off_size + self.len_size],
                                "little")
            if self.filtered:
                raise HDF5Error("Compressed fractal heaps are not supported")
            for boff, baddr, bsize in self.blocks:
                if boff <= off < boff + bsize:
                    start = baddr + (off - boff)
                    return start, ln
            raise HDF5Error("Fractal heap object not found")
        if kind == 2:
            if self.id_len <= 18:
                n = (hid[0] & 0x0F) + 1
                data = hid[1:1 + n]
            else:
                n = (((hid[0] & 0x0F) << 8) | hid[1]) + 1
                data = hid[2:2 + n]
            return bytes(data), n
        raise HDF5Error("'Huge' fractal heap objects are not supported")


def _reshape(flat, shape):
    """Turn a flat list into nested lists of the given shape."""
    if len(shape) <= 1:
        return list(flat)
    step = _prod(shape[1:])
    return [_reshape(flat[i * step:(i + 1) * step], shape[1:]) for i in range(shape[0])]


# --------------------------------------------------------------------------
# Objects
# --------------------------------------------------------------------------

class Node(object):
    kind = "node"

    def __init__(self, file, name, path, addr):
        self.file = file
        self.name = name
        self.path = path
        self.addr = addr
        self._attrs = None

    @property
    def attrs(self):
        """Dictionary of attribute name -> value (decoded lazily)."""
        if self._attrs is None:
            self._attrs = {}
            f = self.file
            for mtype, mflags, data, msize in f.messages(self.addr):
                if mtype == 0x000C:
                    self._read_attr(data)
                elif mtype == 0x0015:
                    self._read_dense_attrs(data)
        return self._attrs

    def _read_attr(self, p):
        f = self.file
        try:
            ver = f.u8(p)
            flags = f.u8(p + 1)
            nsize, tsize, ssize = f.u16(p + 2), f.u16(p + 4), f.u16(p + 6)
            q = p + 8
            if ver == 3:
                q += 1
            name = _decode_text(bytes(f.buf[q:q + nsize]).rstrip(b"\x00"))
            q += _pad8(nsize) if ver == 1 else nsize
            if flags & 1:
                dt = f.shared_datatype(q)
            else:
                dt = f.parse_datatype(q)[0]
            q += _pad8(tsize) if ver == 1 else tsize
            shape = f.parse_dataspace(q)
            q += _pad8(ssize) if ver == 1 else ssize
            if shape is None:
                value = None
            else:
                n = _prod(shape)
                vals = f._decode(dt, bytes(f.buf[q:q + n * dt.size]), n)
                value = vals[0] if shape == () else _reshape(vals, shape)
                if isinstance(value, list) and len(shape) == 1 and n == 1:
                    value = value[0]
            self._attrs[name] = value
        except Exception as e:   # keep going, but show the problem
            self._attrs["<unreadable attribute>"] = str(e)

    def _read_dense_attrs(self, p):
        f = self.file
        flags = f.u8(p + 1)
        q = p + 2
        if flags & 1:
            q += 2
        heap = f.addr(q)
        btree = f.addr(q + f.O)
        if heap is None or btree is None:
            return
        fh = f.fractal_heap(heap)
        for rec in f.btree2_records(btree):
            start, n = fh.get(rec[:fh.id_len if fh.id_len <= 8 else 8])
            if isinstance(start, bytes):
                continue
            self._read_attr(start)

    def __repr__(self):
        return "<%s %r>" % (self.kind, self.path)


class Group(Node):
    kind = "group"

    def __init__(self, *a):
        Node.__init__(self, *a)
        self._links = None

    def links(self):
        """List of (name, kind, target) of the group members."""
        if self._links is not None:
            return self._links
        f = self.file
        links = []
        for mtype, mflags, data, msize in f.messages(self.addr):
            if mtype == 0x0011:
                links.extend(f.symbol_table_links(f.addr(data), f.addr(data + f.O)))
            elif mtype == 0x0006:
                links.append(f.parse_link(data))
            elif mtype == 0x0002:
                lflags = f.u8(data + 1)
                q = data + 2 + (8 if lflags & 1 else 0)
                heap = f.addr(q)
                btree = f.addr(q + f.O)
                if heap is None or btree is None:
                    continue
                fh = f.fractal_heap(heap)
                for rec in f.btree2_records(btree):
                    start, n = fh.get(rec[4:4 + fh.id_len])
                    if not isinstance(start, bytes):
                        links.append(f.parse_link(start))
        links.sort(key=lambda l: natural_key(l[0]))
        self._links = links
        return links

    def keys(self):
        return [l[0] for l in self.links()]

    def items(self):
        out = []
        for name, kind, target in self.links():
            out.append((name, self._resolve(name, kind, target)))
        return out

    def _resolve(self, name, kind, target):
        path = (self.path.rstrip("/") + "/" + name)
        if kind == "hard":
            try:
                return self.file.node_at(target, name, path)
            except Exception as e:
                return BrokenLink(self.file, name, path, None, str(e))
        if kind == "soft":
            try:
                node = self.file.root[target] if target.startswith("/") else self[target]
                return node
            except Exception:
                return BrokenLink(self.file, name, path, None,
                                  "soft link to missing object %s" % target)
        return BrokenLink(self.file, name, path, None, "%s link to %s" % (kind, target))

    def __getitem__(self, path):
        node = self
        for part in [p for p in path.split("/") if p]:
            if not isinstance(node, Group):
                raise KeyError(path)
            for name, kind, target in node.links():
                if name == part:
                    node = node._resolve(name, kind, target)
                    break
            else:
                raise KeyError(path)
        return node

    def __contains__(self, path):
        try:
            self[path]
            return True
        except KeyError:
            return False

    def get(self, path, default=None):
        try:
            return self[path]
        except KeyError:
            return default


class BrokenLink(Node):
    kind = "link"

    def __init__(self, file, name, path, addr, reason):
        Node.__init__(self, file, name, path, addr)
        self.reason = reason
        self._attrs = {}


class CommittedDatatype(Node):
    kind = "datatype"


class Dataset(Node):
    kind = "dataset"

    def __init__(self, *a):
        Node.__init__(self, *a)
        f = self.file
        self.shape = ()
        self.dtype = None
        self._layout = None
        self._filters = []
        for mtype, mflags, data, msize in f.messages(self.addr):
            if mtype == 0x0001:
                self.shape = f.parse_dataspace(data)
            elif mtype == 0x0003:
                self.dtype = f.shared_datatype(data) if mflags & 0x02 else \
                    f.parse_datatype(data)[0]
            elif mtype == 0x0008:
                self._layout = data
            elif mtype == 0x000B:
                self._filters = self._parse_filters(data)

    @property
    def size(self):
        return 0 if self.shape is None else _prod(self.shape)

    @property
    def ndim(self):
        return 0 if not self.shape else len(self.shape)

    def _parse_filters(self, p):
        f = self.file
        ver = f.u8(p)
        n = f.u8(p + 1)
        q = p + (8 if ver == 1 else 2)
        out = []
        for _ in range(n):
            fid = f.u16(q)
            q += 2
            nlen = 0
            if ver == 1 or fid >= 256:
                nlen = f.u16(q)
                q += 2
            q += 2      # flags
            nvals = f.u16(q)
            q += 2
            q += _pad8(nlen) if ver == 1 else nlen
            vals = [f.u32(q + 4 * i) for i in range(nvals)]
            q += 4 * nvals
            if ver == 1 and nvals % 2:
                q += 4
            out.append((fid, vals))
        return out

    def raw_bytes(self):
        """Return the raw bytes of the whole dataset."""
        f = self.file
        p = self._layout
        esize = self.dtype.size
        total = self.size * esize
        ver = f.u8(p)
        if ver in (1, 2):
            nd = f.u8(p + 1)
            cls = f.u8(p + 2)
            q = p + 8
            address = None
            if cls != 0:
                address = f.addr(q)
                q += f.O
            dims = [f.u32(q + 4 * i) for i in range(nd)]
            q += 4 * nd
            if cls == 0:
                n = f.u32(q)
                return bytes(f.buf[q + 4:q + 4 + n])
            if cls == 1:
                return self._contiguous(address, total)
            return self._chunked_btree1(address, dims[:-1] if len(dims) > len(self.shape) else dims, esize)
        cls = f.u8(p + 1)
        q = p + 2
        if cls == 0:
            n = f.u16(q)
            return bytes(f.buf[q + 2:q + 2 + n])
        if cls == 1:
            return self._contiguous(f.addr(q), total)
        if cls == 2 and ver == 3:
            nd = f.u8(q)
            address = f.addr(q + 1)
            dims = [f.u32(q + 1 + f.O + 4 * i) for i in range(nd)]
            return self._chunked_btree1(address, dims[:-1], esize)
        if cls == 2 and ver >= 4:
            flags = f.u8(q)
            nd = f.u8(q + 1)
            dsize = f.u8(q + 2)
            q += 3
            dims = [f.uint(q + dsize * i, dsize) for i in range(nd)]
            q += dsize * nd
            itype = f.u8(q)
            q += 1
            chunk = dims[:-1]
            if itype == 1:            # single chunk
                fsize, mask = None, 0
                if flags & 0x02:
                    fsize = f.length(q)
                    mask = f.u32(q + f.L)
                    q += f.L + 4
                address = f.addr(q)
                if address is None:
                    return bytes(total)
                n = fsize if fsize is not None else _prod(chunk) * esize
                data = self._unfilter(bytes(f.buf[address:address + n]), mask, esize)
                out = bytearray(total)
                self._place(out, data, [0] * len(chunk), chunk, esize)
                return bytes(out)
            if itype == 2:            # implicit: chunks stored in order
                address = f.addr(q)
                out = bytearray(total)
                if address is None:
                    return bytes(out)
                csize = _prod(chunk) * esize
                grid = [-(-s // c) for s, c in zip(self.shape, chunk)]
                for k, idx in enumerate(itertools.product(*[range(g) for g in grid])):
                    a = address + k * csize
                    self._place(out, bytes(f.buf[a:a + csize]),
                                [i * c for i, c in zip(idx, chunk)], chunk, esize)
                return bytes(out)
            if itype == 3:            # fixed array
                q += 1                # page bits
                return self._chunked_fixed_array(f.addr(q), chunk, esize, flags)
            raise HDF5Error(
                "This dataset uses a chunk index (type %d) written with "
                "HDF5 'latest format' settings, which this reader does not "
                "support yet." % itype)
        raise HDF5Error("Unsupported data layout (version %d, class %d)" % (ver, cls))

    def _contiguous(self, address, total):
        if address is None:
            return bytes(total)
        return bytes(self.file.buf[address:address + total])

    def _unfilter(self, data, mask, esize):
        for j in range(len(self._filters) - 1, -1, -1):
            if mask & (1 << j):
                continue
            fid, vals = self._filters[j]
            if fid == 1:
                data = zlib.decompress(data)
            elif fid == 2:
                es = vals[0] if vals else esize
                n = len(data) // es
                out = bytearray(data)
                for b in range(es):
                    out[b:n * es:es] = data[b * n:(b + 1) * n]
                data = bytes(out)
            elif fid == 3:
                data = data[:-4]
            else:
                names = {4: "szip", 32000: "lzf", 32001: "blosc", 32004: "lz4",
                         32015: "zstd"}
                raise HDF5Error("Compression filter '%s' is not supported "
                                "(only gzip/shuffle/fletcher32)"
                                % names.get(fid, fid))
        return data

    def _place(self, out, data, offset, chunk, esize):
        shape = self.shape
        nd = len(shape)
        if nd == 0:
            out[:esize] = data[:esize]
            return
        ext = [min(c, s - o) for c, s, o in zip(chunk, shape, offset)]
        if any(e <= 0 for e in ext):
            return
        run = ext[-1] * esize
        cstr = [0] * nd
        sstr = [0] * nd
        c = s = 1
        for i in range(nd - 1, -1, -1):
            cstr[i] = c
            sstr[i] = s
            c *= chunk[i]
            s *= shape[i]
        for idx in itertools.product(*[range(e) for e in ext[:-1]]):
            src = sum(i * st for i, st in zip(idx, cstr)) * esize
            dst = sum((i + o) * st for i, o, st in zip(idx, offset, sstr))
            dst = (dst + offset[-1]) * esize
            out[dst:dst + run] = data[src:src + run]

    def _chunk_grid(self, chunk):
        grid = [-(-s // c) for s, c in zip(self.shape, chunk)]
        return list(itertools.product(*[range(g) for g in grid]))

    def _chunked_fixed_array(self, address, chunk, esize, flags):
        f = self.file
        out = bytearray(self.size * esize)
        if address is None:
            return bytes(out)
        f._check_sig(address, b"FAHD")
        client = f.u8(address + 5)
        entry = f.u8(address + 6)
        page_bits = f.u8(address + 7)
        nelem = f.length(address + 8)
        dblock = f.addr(address + 8 + f.L)
        if dblock is None:
            return bytes(out)
        f._check_sig(dblock, b"FADB")
        p = dblock + 6 + f.O
        page = 1 << page_bits
        positions = []
        if nelem > page:
            npages = -(-nelem // page)
            p += -(-npages // 8) + 4          # bitmap + checksum
            for i in range(nelem):
                pg, k = divmod(i, page)
                positions.append(p + pg * (page * entry + 4) + k * entry)
        else:
            positions = [p + i * entry for i in range(nelem)]
        csize_bytes = entry - f.O - 4
        full = _prod(chunk) * esize
        for idx, pos in zip(self._chunk_grid(chunk), positions):
            a = f.addr(pos)
            if a is None:
                continue
            if client == 1:
                n = f.uint(pos + f.O, csize_bytes)
                mask = f.u32(pos + f.O + csize_bytes)
            else:
                n, mask = full, 0
            data = self._unfilter(bytes(f.buf[a:a + n]), mask, esize)
            self._place(out, data, [i * c for i, c in zip(idx, chunk)], chunk, esize)
        return bytes(out)

    def _chunked_btree1(self, address, chunk, esize):
        f = self.file
        out = bytearray(self.size * esize)
        if address is None:
            return bytes(out)
        nd = len(chunk) + 1
        ksize = 8 + 8 * nd

        def walk(node):
            f._check_sig(node, b"TREE")
            level = f.u8(node + 5)
            n = f.u16(node + 6)
            p = node + 8 + 2 * f.O
            for _ in range(n):
                csize = f.u32(p)
                mask = f.u32(p + 4)
                offs = [f.u64(p + 8 + 8 * j) for j in range(nd - 1)]
                child = f.addr(p + ksize)
                p += ksize + f.O
                if level > 0:
                    walk(child)
                else:
                    data = self._unfilter(bytes(f.buf[child:child + csize]), mask, esize)
                    self._place(out, data, offs, chunk, esize)
        walk(address)
        return bytes(out)

    def read(self):
        """Return all values as a flat list (row-major order)."""
        if self.shape is None:
            return []
        n = self.size
        return self.file._decode(self.dtype, self.raw_bytes(), n)

    def value(self):
        """Return the data as nested lists (or a single value for scalars)."""
        flat = self.read()
        if self.shape == ():
            return flat[0] if flat else None
        return _reshape(flat, self.shape)


def natural_key(s):
    """Sort key so that 'tally 2' comes before 'tally 10'."""
    out = []
    num = ""
    for ch in s:
        if ch.isdigit():
            num += ch
        else:
            if num:
                out.append((0, int(num), ""))
                num = ""
            out.append((1, 0, ch.lower()))
    if num:
        out.append((0, int(num), ""))
    return out
