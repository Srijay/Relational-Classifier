import gzip, struct


class SeqzFileReader(object):
    def __init__(self, path):
        self.gzf = gzip.open(path, 'rb')

    def getBytes(self, lena):
        ans = bytearray()
        while len(ans) < lena:
            next = self.gzf.read(lena-len(ans))
            if len(next) == 0: break;
            ans += next
        return ans

    def bytesToStr(self, ans):
        ret = str()
        for bb in ans:
            ret += chr(bb)
        return ret

    def get(self):
        """
        :return: Next record as string, empty if eof.
        """
        lenlen = struct.calcsize("i")
        len1str = self.getBytes(lenlen)
        if len(len1str) == 0: return str()
        len1, = struct.unpack("i", len1str)
        rec = str(self.getBytes(len1)) # self.bytesToStr(self.getBytes(len1))
        len2, = struct.unpack("i", self.getBytes(lenlen))
        assert len1 == len2, str(len1) + "!=" + str(len2)
        return rec

    def close(self):
        self.gzf.close()


class SeqzFileWriter(object):
    def __init__(self, path):
        self.gzf = gzip.open(path, 'wb')

    def put(self, rec):
        """
        :param rec: Record to write, as byte sequence.
        :return: nothing
        """
        lenlen = struct.calcsize("i")
        len1 = len(rec)
        lenstr = struct.pack("i", len1)
        wlen1 = self.gzf.write(lenstr)
        assert wlen1 == lenlen
        wlen2 = self.gzf.write(rec)
        assert wlen2 == len1, str(wlen2) + " != " + str(len1)
        wlen3 = self.gzf.write(lenstr)
        assert wlen3 == lenlen

    def close(self):
        self.gzf.close()

# Test harness.
if __name__ == "__main__":
    fname = '/tmp/seqzfile'
    recs = ['\x9f\x9e\xaa', 'abc', 'defg', '12345', '!@#$%^']
    sfw = SeqzFileWriter(fname)
    for rec in recs:
        sfw.put(rec)
    sfw.close()
    sfr = SeqzFileReader(fname)
    numRec = 0
    while True:
        rec = sfr.get()
        if rec == '': break
        assert rec == recs[numRec]
        numRec += 1
