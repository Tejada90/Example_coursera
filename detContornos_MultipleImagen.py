import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt
import pandas as pd

font = cv2.FONT_HERSHEY_SIMPLEX
def getKey(item):
    return item[0]

cap = cv2.VideoCapture("ejemplo2.mp4")
fourcc = cv2.VideoWriter_fourcc(*'DIVX')
out = cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480))
cntFrame = 0;
vAcum = []
vAcum2 = []
d = 33.16743834924821
dr = 20
r = dr/d

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret==True:
        ejemploGris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret,ejemploBin_2 = cv2.threshold(ejemploGris, 150, 255,  cv2.THRESH_BINARY)
        kernel = np.ones((5,5),np.uint8)
        ejemploBin_2 = cv2.morphologyEx(ejemploBin_2,cv2.MORPH_OPEN,kernel)
        ejemploBin_2 = cv2.morphologyEx(ejemploBin_2,cv2.MORPH_CLOSE,kernel)
        #cv2.imshow("ejemploBin_2", ejemploBin_2)
        cnts, her = cv2.findContours(ejemploBin_2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        vCont=[]

        for cnt in cnts:
            area_cnt = cv2.contourArea(cnt)
            momentos = cv2.moments(cnt)
            cx = int(momentos['m10']/momentos['m00'])
            cy = int(momentos['m01']/momentos['m00'])
            approx = cv2.approxPolyDP(cnt, 0.01* cv2.arcLength(cnt, True), True)
            if len(approx) == 4 :
                xRef=cx
                yRef=cy
                #print(xRef)
                #print(yRef)
                        
        for cnt in cnts:
            area_cnt = cv2.contourArea(cnt)
            momentos = cv2.moments(cnt)
            cx = int(momentos['m10']/momentos['m00'])
            cy = int(momentos['m01']/momentos['m00'])
                
            #print(cnt)
            #cv2.drawContours(frame, cnt, -1, (0,255,0), 2)
                            
            if (area_cnt < 3500 and area_cnt > 400):
                approx = cv2.approxPolyDP(cnt, 0.01* cv2.arcLength(cnt, True), True)
                cv2.drawContours(frame, cnt, -1, (0,255,0), 2)
                #print(cx)
                #print(cy)
                if len(approx) == 4 :
                    x, y , w, h = cv2.boundingRect(approx)
                    aspectRatio = float(w)/h
                    #print(aspectRatio)
                    if aspectRatio >= 0.95 and aspectRatio < 1.05:
                        cv2.putText(frame, "square", (cx, cy), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))
                        cv2.putText(frame, str(her[cnt,1]), (her[cnt,1]), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))

                    else:
                        cxN=cx-xRef
                        cyN=cy-yRef
                        vCont.append([cxN*r,cyN*r])
                        cv2.putText(frame, "rectangle", (cx, cy+50), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))
                        #cv2.putText(frame, str(her[cnt,1]), (her[cnt,1]), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))

                else:
                    cxN=-cx+xRef
                    cyN=-cy+yRef
                    vCont.append([cxN*r,cyN*r])
                    cv2.putText(frame, "circle", (cx, cy+50), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))
                    #cv2.putText(frame, str(her[cnt,1]), (her[cnt,1]), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))

    #Dibujar el centro
                cv2.circle(frame,(cx, cy), 3, (0,0,255), 1)
               
    #Escribimos las coordenadas del centro
                cv2.putText(frame,"(x: " + str(cxN*r) + ", y: " + str(cyN*r) + ")",(cx-50, cy-50), font, 0.5,(255,255,255),1)
#Mostramos la imagen final
##            cv2.imshow("Ejemplo_Contornos", frame)#cv2.imshow('frame',frame)
            #out.write(frame)
        if cv2.waitKey(1) == ord('q'):
            break

        vCont = sorted(vCont, key=getKey)
        vAcum.append(vCont)
    else:
        break
#Libera todo si la tarea ha terminado

nit= len(vAcum)

##for i in range(nit):
##    vAcum2 = [(elem1, elem2) for elem1, elem2 in vAcum[i]]
##    plt.cla()
##    plt.plot(*zip(*vAcum2))
##    plt.scatter(*zip(*vAcum2))
##    plt.xlim(-700*r, 700*r)
##    plt.ylim(-700*r, 700*r)
##    plt.pause(0.00001)
##
##plt.show()
my_df = pd.DataFrame(vAcum)
my_df.to_excel('out.xlsx',header=False, index=False )


cap.release()
cv2.destroyAllWindows()
