import numpy as np
import cv2

class ColorGradientPoint(object):
    def __init__(self, p_r, p_g, p_b, p_v):
        self._r, self._g, self._b = p_r, p_g, p_b
        self._v = p_v; #position of the color along the gradient (0 to 1)

class ColorGradient(object):
    def __init__(self):
        self._colorGradientPtArr = []

    def AddGradientColorPoint(self, p_r, p_g, p_b, p_v):
        colorPoint = ColorGradientPoint(p_r, p_g, p_b, p_v)
        for ci in range(len(self._colorGradientPtArr)):
            c = self._colorGradientPtArr[ci]
            if p_v < c._val:
                self._colorGradientPtArr.insert(ci, colorPoint)
                return
        self._colorGradientPtArr.Append(colorPoint)

    def ClearGradient(self):
        self._colorGradientPtArr = []

    def DefaultGradientCx(self):        
        self._colorGradientPtArr = [
            ColorGradientPoint(0,0,1, 0.0), #blue
            ColorGradientPoint(0,1,1, 0.25), #cyan
            ColorGradientPoint(0,1,0, 0.5), #green
            ColorGradientPoint(1,1,0, 0.75), #yellow
            ColorGradientPoint(1,0,0, 1.0), #red
        ]
    
    def GradientColorAtValueRx(self, p_v, p_heatImg):

        colorGradientPointArrLen = len(self._colorGradientPtArr)
        if colorGradientPointArrLen <= 0:
            return (r,g,b)

        heatImgMask = np.ones_like(p_v, dtype=np.bool)

        for ci in range(colorGradientPointArrLen):
            colorGradientPt = self._colorGradientPtArr[ci]              #current gradient color pt
            colorGradientPt0 = self._colorGradientPtArr[max(0,ci-1)]    #previous gradient color pt
            dv = (colorGradientPt0._v - colorGradientPt._v)

            colorGradientPtIdx = p_v < colorGradientPt._v

            f = np.zeros_like(p_v) if dv == 0.0 else (p_v - colorGradientPt._v) / dv

            r = (colorGradientPt0._r - colorGradientPt._r) * f + colorGradientPt._r
            g = (colorGradientPt0._g - colorGradientPt._g) * f + colorGradientPt._g
            b = (colorGradientPt0._b - colorGradientPt._b) * f + colorGradientPt._b

            p_heatImg[:,:,2][heatImgMask] = r[heatImgMask]
            p_heatImg[:,:,1][heatImgMask] = g[heatImgMask]
            p_heatImg[:,:,0][heatImgMask] = b[heatImgMask]

            heatImgMask[colorGradientPtIdx] = False


def heatmap0(p_img, p_imgMean):
    imgMeanFactor = np.zeros_like(p_imgMean, dtype=np.float)
    bgIdx = p_imgMean <= 127
    grIdx = p_imgMean > 127
    imgMeanFactor[bgIdx] = p_imgMean[bgIdx]/127.0
    imgMeanFactor[grIdx] = (255-p_imgMean[grIdx])/127.0
    imgHeat = np.zeros_like(p_img)
    #b,g,r
    imgHeat[bgIdx,0] = (1-imgMeanFactor[bgIdx])*255
    imgHeat[bgIdx,1] = imgMeanFactor[bgIdx]*255
    imgHeat[grIdx,2] = (1-imgMeanFactor[grIdx])*255
    imgHeat[grIdx,1] = imgMeanFactor[grIdx]*255
    cv2.imwrite('./Out/imgHeat.png', imgHeat)


imgPath = './In/Swan.jpg'
img = cv2.imread(imgPath, cv2.IMREAD_UNCHANGED)

imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imwrite('./Out/imgGray.png', imgGray)

img = img/255.0
imgGray = imgGray/255.0

colorGradient = ColorGradient()
colorGradient.DefaultGradientCx();

imgHeat = np.zeros_like(img)
colorGradient.GradientColorAtValueRx(imgGray, imgHeat);
imgHeat = cv2.normalize(imgHeat, imgHeat, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
cv2.imwrite('./Out/imgHeat.png', imgHeat)