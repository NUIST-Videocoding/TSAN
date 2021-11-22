import cv2
import numpy as np

def imread(image_path, is_grayscale=False):
    """read image: (H, W, C), in which the order of Channel is RGB"""
    # if image is yuv, then call yuvread
    if is_grayscale:
        img = cv2.imread(image_path, 0)
        img = np.expand_dims(img, axis=-1)
    else:
        img = cv2.imread(image_path)
    img[:, :, :] = img[:, :, [2, 1, 0]]  # change from BGR to RGB
    return img

def imwrite(img, image_path):
    """write img to image_path"""
    img[:, :, :] = img[:, :, [2, 1, 0]]  # change from RGB to BGR
    cv2.imwrite(image_path, img)

class YuvReader(object):
    def __init__(self, yuv_path, y_h=None, y_w=None, v_format='yuv420'):
        self.yuv_path = yuv_path
        self.v_format = v_format
        self.yuv_handle = open(yuv_path, 'rb')
        self.y_w, self.y_h = y_w, y_h
        self.y_size = self.y_w * self.y_h

    def read(self, n_frames, yuv_is_tuple=False):
        """ read n_frames, which format is
        (N, H, W, C) if yuv_is_tuple is False,and C is 3 when yuv444,yuv420, C is 1 when yuv400
        else ((N,H,W,1),(N,H,W,1),(N,H,W,1))"""
        yuv = None
        if self.v_format == 'yuv444':
            yuv = self._read_yuv444(n_frames, yuv_is_tuple)

        elif self.v_format == 'yuv420':
            yuv = self._read_yuv420(n_frames, yuv_is_tuple)
        elif self.v_format == 'yuv400':
            yuv = self._read_yuv400(n_frames, yuv_is_tuple)
        return yuv

    def read_done(self):
        self.yuv_handle.close()

    def _read_with_exception(self, size):
        raw_data = self.yuv_handle.read(size)
        if len(raw_data) < size:
            raise _NoMoreDataException('no more data')
        data = list(map(int, raw_data))
        return data

    def _read_yuv444(self, n_frames, yuv_is_tuple):
        self.u_w = self.y_w
        self.u_h = self.y_h
        self.u_size = self.u_w * self.u_h

        data_y, data_u, data_v = [], [], []
        for idx in range(n_frames):
            try:
                single_y = self._read_with_exception(self.y_size)
                single_u = self._read_with_exception(self.u_size)
                single_v = self._read_with_exception(self.u_size)
            except _NoMoreDataException:
                if len(data_y) == 0:
                    return [None, None, None] if yuv_is_tuple else None
                else:
                    break
            else:
                single_y = np.reshape(single_y, (1, self.y_h, self.y_w, 1))
                single_u = np.reshape(single_u, (1, self.u_h, self.u_w, 1))
                single_v = np.reshape(single_v, (1, self.u_h, self.u_w, 1))
                data_y.append(single_y)
                data_u.append(single_u)
                data_v.append(single_v)
        data_y = np.concatenate(data_y, axis=0)
        data_u = np.concatenate(data_u, axis=0)
        data_v = np.concatenate(data_v, axis=0)
        rets = [data_y, data_u, data_v]
        if yuv_is_tuple:
            return rets
        else:
            return np.concatenate(rets, axis=-1)

    def _read_yuv420(self, n_frames, yuv_is_tuple):
        self.u_w = self.y_w // 2
        self.u_h = self.y_h // 2
        self.u_size = self.u_w * self.u_h

        data_y, data_u, data_v = [], [], []
        for idx in range(n_frames):
            try:
                single_y = self._read_with_exception(self.y_size)
                single_u = self._read_with_exception(self.u_size)
                single_v = self._read_with_exception(self.u_size)
            except _NoMoreDataException:
                if len(data_y) == 0:
                    return [None, None, None] if yuv_is_tuple else None
                else:
                    break
            else:
                single_y = np.reshape(single_y, (1, self.y_h, self.y_w, 1))
                single_u = np.reshape(single_u, (1, self.u_h, self.u_w, 1))
                single_v = np.reshape(single_v, (1, self.u_h, self.u_w, 1))
                data_y.append(single_y)
                data_u.append(single_u)
                data_v.append(single_v)
        data_y = np.concatenate(data_y, axis=0)
        data_u = np.concatenate(data_u, axis=0)
        data_v = np.concatenate(data_v, axis=0)

        if yuv_is_tuple:
            return [data_y, data_u, data_v]
        else:
            data_u = upsample_chroma(data_u)
            data_v = upsample_chroma(data_v)
            return np.concatenate([data_y, data_u, data_v], axis=-1)

    def _read_yuv400(self, n_frames, yuv_is_tuple):
        self.u_w = self.u_h = self.u_size = 0

        data_y = []
        for idx in range(n_frames):
            try:
                single_y = self._read_with_exception(self.y_size)
            except _NoMoreDataException:
                if len(data_y) == 0:
                    return [None, None, None] if yuv_is_tuple else None
                else:
                    break
            else:
                single_y = np.reshape(single_y, (1, self.y_h, self.y_w, 1))
                data_y.append(single_y)
        data_y = np.concatenate(data_y, axis=0)
        rets = [data_y, None, None]
        if yuv_is_tuple:
            return rets
        else:
            return data_y


class YuvWriter(object):
    def __init__(self, yuv_path):
        self.yuv_path = yuv_path
        self.yuv_handle = open(yuv_path, 'wb')

    def write_done(self):
        self.yuv_handle.close()

    def write(self, yuv_data, yuv_is_tuple):
        if yuv_is_tuple:
            if yuv_data[1] is None or yuv_data[2] is None:
                # yuv400
                flatten_data = yuv_data[0].astype('uint8').ravel()
                stream_data = ''.join(map(chr, flatten_data)).encode(encoding='ascii')
                self.yuv_handle.write(stream_data)
            else:
                if yuv_data[0].ndim == 3:
                    yuv_data = list(map(lambda x: x[None, ...], yuv_data))

                assert yuv_data[0].shape[0] == yuv_data[1].shape[0] and \
                       yuv_data[1].shape[0] == yuv_data[2].shape[0]

                for frame_i in range(yuv_data[0].shape[0]):
                    for yuv_j in range(3):
                        data_i_j = yuv_data[yuv_j][frame_i]
                        flatten_data = data_i_j.astype('uint8').ravel()
                        stream_data = ''.join(map(chr, flatten_data)).encode()
                        self.yuv_handle.write(stream_data)
        else:
            self._write_nhwc_hwc(yuv_data)

    def _write_nhwc_hwc(self, data):
        dim_h = data.ndim - 3
        dim_w = data.ndim - 2
        dim_c = data.ndim - 1
        transposed_data = np.swapaxes(data, dim_h, dim_c)
        transposed_data = np.swapaxes(transposed_data, dim_h, dim_w)
        flatten_data = transposed_data.astype('uint8').ravel()
        stream_data = ''.join(map(chr, flatten_data)).encode()
        self.yuv_handle.write(stream_data)


class _NoMoreDataException(Exception):
    def __init__(self, msg):
        self.msg = msg

    def __str__(self):
        return repr(self.msg)


def tiny_yuvread(yuv_path, y_h=None, y_w=None, n_frames=None, v_format='yuv420'):
    """read one yuv file"""
    reader = YuvReader(yuv_path, y_h, y_w, v_format=v_format)
    yuv = reader.read(n_frames, yuv_is_tuple=False)
    reader.read_done()
    return yuv


def upsample_chroma(chroma_nhwc):
    """upsample from (H, W, C) to (2H, 2W, C), or from (N, H, W, C) to (N, 2H, 2W, C)"""
    ndim = chroma_nhwc.ndim
    dim_h = ndim - 3
    dim_w = ndim - 2
    upsampled_chroma = np.repeat(chroma_nhwc, 2, axis=dim_h)
    upsampled_chroma = np.repeat(upsampled_chroma, 2, axis=dim_w)
    return upsampled_chroma



if __name__ == '__main__':
    path = "./BasketballPass_416x240_50_444.yuv"
    h = 416
    w = 240
    yuvRead = YuvReader(path,h,w,'yuv444')
    yuv = yuvRead.read(2)

    print(yuv.shape)
