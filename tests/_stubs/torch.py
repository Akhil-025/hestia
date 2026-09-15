class _Cuda:
    @staticmethod
    def is_available(): return False
    @staticmethod
    def set_device(i): pass
cuda = _Cuda()
