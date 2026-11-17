"""Read selected official ZIP members over validated HTTP byte ranges.

Downloads one real RGB-D frame and its rectangle labels, not the full dataset.
"""
import io
import json
import hashlib
from pathlib import Path
from urllib.request import Request, urlopen
from zipfile import ZipFile


class RemoteZip(io.RawIOBase):
    def __init__(self, file_id):
        self.url = ('https://drive.usercontent.google.com/download?export=download'
                    '&confirm=t&id=' + file_id)
        self.pos = 0
        self.size = None
        self.fetch(0, 0)

    def fetch(self, start, end):
        # Unique range query avoids intermediary caches returning another range.
        req = Request(self.url + '&byte_range=%d-%d' % (start, end),
                      headers={'Range': 'bytes=%d-%d' % (start, end)})
        with urlopen(req, timeout=60) as r:
            cr = r.headers.get('Content-Range', '')
            if r.status != 206 or not cr.startswith('bytes %d-' % start):
                raise RuntimeError('Server did not honor byte range %d-%d: %s %s %s' % (start, end, r.status, cr, r.read(512)))
            self.size = int(cr.split('/')[-1])
            data = r.read(end - start + 2)
        if len(data) != end - start + 1:
            raise RuntimeError('Incomplete or oversized range response')
        return data

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        self.pos = offset + (0 if whence == 0 else self.pos if whence == 1 else self.size)
        if self.pos < 0:
            raise ValueError('Negative seek')
        return self.pos

    def read(self, n=-1):
        n = self.size - self.pos if n < 0 else min(n, self.size - self.pos)
        if n <= 0:
            return b''
        data = self.fetch(self.pos, self.pos + n - 1)
        self.pos += len(data)
        return data


def main():
    root = Path(__file__).resolve().parents[1]
    records = []
    sources = [
        ('train_1.zip', '1wQx8IJ_Lok3hVK_nchQUzgw88QBGZ5iq',
         ['scene_0000/realsense/rgb/0000.png', 'scene_0000/realsense/depth/0000.png']),
        ('rect_labels.zip', '1lR6ZSgtgV1KlqzM14mKlQ8oKhE3UCltO',
         ['scene_0000/realsense/rect/0000.npy']),
    ]
    for archive, fid, suffixes in sources:
        with ZipFile(RemoteZip(fid)) as z:
            names = z.namelist()
            for suffix in suffixes:
                matches = [n for n in names if n.endswith(suffix)]
                if len(matches) != 1:
                    raise RuntimeError('Expected one %s; found %s. Examples: %s' %
                                       (suffix, matches, names[:10]))
                payload = z.read(matches[0])  # ZIP CRC verified by zipfile.
                dest = root / 'data' / 'scenes' / suffix
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(payload)
                records.append(dict(archive=archive, file_id=fid, member=matches[0],
                                    path=str(dest.relative_to(root)), bytes=len(payload),
                                    sha256=hashlib.sha256(payload).hexdigest()))
                print('Downloaded', suffix, len(payload), flush=True)
                (root / 'audit' / 'sample_download.json').write_text(json.dumps(records, indent=2))


if __name__ == '__main__':
    main()
