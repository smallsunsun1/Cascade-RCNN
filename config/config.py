import easydict

config = easydict.EasyDict()
_C = config
_C.MODE_MASK =True
_C.MODE_FPN = False

_C.RPN = easydict.EasyDict()
_C.RPN.ANCHOR_STRIDE = 16
_C.RPN.ANCHOR_SIZES = (32, 64, 128, 256, 512)   # sqrtarea of the anchor box