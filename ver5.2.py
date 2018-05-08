# -*- coding: utf-8 -*-
"""
Created on Tue Oct 1 16:06:10 2017

@author: Kien Nguyen
New:
    - Update function
"""
import numpy as np
import matplotlib
import tkinter.messagebox as mbox
from tkinter import *
from scipy.io import wavfile
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from tkinter.ttk import Button

class Window(Frame):
    def __init__(self, parent):
        Frame.__init__(self, parent)
        self.parent = parent
        self.initUI()
        self.fileName = ''
        self.ipFig = Figure(dpi=50)
        self.ipAx = self.ipFig.add_subplot(111)
        self.sylFig = Figure(dpi=50)
        self.sylAx = self.sylFig.add_subplot(111)
        self.f0Fig = Figure(dpi=50)
        self.f0Ax = self.f0Fig.add_subplot(111)
        self.plotted = False 
        self.sylPlotted = False
        self.f0Plotted = False
        self.ipCanvas = FigureCanvasTkAgg(self.ipFig, self.ipAxesFrame)   
        self.sylCanvas = FigureCanvasTkAgg(self.sylFig, self.sylAxesFrame)
        self.f0Canvas = FigureCanvasTkAgg(self.f0Fig, self.f0AxesFrame)
        self.sylList = []

    def initUI(self):
        self.parent.title('GR2')
        self.pack(fill=BOTH, expand=True)
        self.opMenuVar = StringVar(root)        
        
        self.fnFrame = Frame(self)
        self.fnFrame.pack(side=LEFT, fill=Y, anchor=NW)

        basicFnFrame = Frame(self.fnFrame, borderwidth=2, relief='groove')
        basicFnFrame.pack(ipady=1)
        brFrame = Frame(basicFnFrame)
        brFrame.pack(fill=X, pady=3)
        Button(brFrame, text='Chọn file1', command=self.browseFile1).pack(side=LEFT, padx=22)
        self.pathLabel = Label(brFrame, text='', borderwidth=2, relief='groove', width=17)
        self.pathLabel.pack(fill=X, side=LEFT, padx=5)

        basicFnFrame = Frame(self.fnFrame, borderwidth=2, relief='groove')
        basicFnFrame.pack(ipady=1)
        brFrame = Frame(basicFnFrame)
        brFrame.pack(fill=X, pady=3)
        Button(brFrame, text='Chọn file2', command=self.browseFile2).pack(side=LEFT, padx=22)
        self.pathLabe2 = Label(brFrame, text='', borderwidth=2, relief='groove', width=17)
        self.pathLabe2.pack(fill=X, side=LEFT, padx=5)

        basicFnFrame = Frame(self.fnFrame, borderwidth=2, relief='groove')
        basicFnFrame.pack(ipady=1)
        brFrame = Frame(basicFnFrame)
        brFrame.pack(fill=X, pady=3)
        Button(brFrame, text='Chọn file3', command=self.browseFile3).pack(side=LEFT, padx=22)
        self.pathLabe3 = Label(brFrame, text='', borderwidth=2, relief='groove', width=17)
        self.pathLabe3.pack(fill=X, side=LEFT, padx=5)

        esFrame = Frame(basicFnFrame)
        esFrame.pack(fill=X)
        Label(esFrame, text='Ngưỡng năng lượng', width=17).pack(side=LEFT)
        self.esEntry = Entry(esFrame, width=13)
        self.esEntry.pack(side=LEFT)
             
        frmFrame = Frame(basicFnFrame)
        frmFrame.pack(fill=X, pady=3)
        Label(frmFrame, text='Độ dài frame', width=17).pack(side=LEFT)
        self.frmEntry = Entry(frmFrame, width=13)
        self.frmEntry.pack(side=LEFT)
        Label(frmFrame, text='(mẫu)').pack(side=LEFT)
        
        timeFrame = Frame(basicFnFrame)
        timeFrame.pack(fill=X)
        Label(timeFrame, text='Ngưỡng thời gian', width=17).pack(side=LEFT)
        self.timeEntry = Entry(timeFrame, width=13)
        self.timeEntry.pack(side=LEFT)
        Label(timeFrame, text='(ms)').pack(side=LEFT)
        
        btnFrame = Frame(basicFnFrame)
        btnFrame.pack(fill=X, pady=3)
        Button(btnFrame, text='Xử lý', command=self.sylMark).pack(side=LEFT, padx=22)
        Button(btnFrame, text='Nghe', command=self.playIp).pack(side=LEFT, padx=4)
        
        Frame(self.fnFrame).pack(pady=118)
        
        self.sylFnFrame = None
        
        axesFrame = Frame(self)
        axesFrame.pack(side=LEFT, anchor=N, fill=BOTH, expand=True) 
           
        self.ipAxesFrame = Frame(axesFrame, borderwidth=2, relief='groove', height=375)
        self.ipAxesFrame.pack(fill=X)
        self.ipAxesFrame.pack_propagate(0)
        
        self.sylAxesFrame = Frame(axesFrame, borderwidth=2, relief='groove', width=550)
        self.sylAxesFrame.pack(side=LEFT, fill=BOTH)
        self.sylAxesFrame.pack_propagate(0)
        
        self.f0AxesFrame = Frame(axesFrame, borderwidth=2, relief='groove', width=1200)
        self.f0AxesFrame.pack(side=LEFT, fill=BOTH)
        self.f0AxesFrame.pack_propagate(0)
        
        matplotlib.rcParams.update({'font.size': 20})

        self.audio1 = []
        self.audio2 = []
        self.audio3 = []

        # tạo mảng rỗng 3D
        n=14
        m=2000
        self.inputPCA = [[0 for k in range(m)] for i in range(n)]
        self.inputPCA = np.asanyarray(self.inputPCA)
        self.inputPCA = self.inputPCA.transpose()
        print(self.inputPCA.shape)
        self.target = []

    def initSylUI(self):
        self.sylFnFrame = Frame(self.fnFrame, borderwidth=2, relief='groove')
        self.sylFnFrame.pack(fill=BOTH, pady=1, ipady=3)
        splitSylFrame = Frame(self.sylFnFrame)
        splitSylFrame.pack(pady=3)
        Label(splitSylFrame, text='Tìm thấy %d âm' %(len(self.sylList))).pack(pady=2)
        Label(splitSylFrame, text='Tách âm:').pack(side=LEFT)
        OptionMenu(splitSylFrame, self.opMenuVar, *self.opMenuChoices, command=self.splitSyl).pack(side=LEFT, padx=10)
        
        hammingFrame = Frame(self.sylFnFrame)
        hammingFrame.pack()

        Button(hammingFrame, text='Nghe', command=self.playSyl).pack(side=LEFT, padx=10)
        # Button(hammingFrame, text='Hamming', command=self.plotHamming).pack(side=LEFT, padx=8)
        Button(hammingFrame, text='PCA,LDA', command=self.PCALDA).pack(side=LEFT, padx=8)

    def PCALDA(self):
        import matplotlib.pyplot as plt
        from sklearn.decomposition import PCA
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        # languages = ["anh", "nhat", "viet"]
        X = self.inputPCA[0:600]
        X = X.astype(np.float)
        np.savetxt('matrix.txt', X, fmt='%.2f')
        a = self.target[0:200]
        print(a)
        b = self.target[len(self.audio1):len(self.audio1)+200]
        print(b)
        c = self.target[len(self.audio1)+len(self.audio2):len(self.audio1)+len(self.audio2)+200]
        print(c)
        y = np.concatenate((a,b,c))
        print(len(y))
        print(y)
        print(X.shape)
        pca = PCA(n_components=2)
        X_r = pca.fit(X).transform(X)

        # Percentage of variance explained for each components
        print('explained variance ratio (first two components): %s'
              % str(pca.explained_variance_ratio_))

        plt.figure()
        # plt.subplot(211)
        # colors = ['navy', 'turquoise', 'darkorange']
        # lw = 2
        print(X_r)
        # for color,target_name in zip(colors, languages):
        #     plt.scatter(color=color, alpha=.8, lw=lw,label=target_name)
        plt.title('PCA of dataset')
        plt.plot(X_r[0:200, 0], X_r[0:200, 1], 'ro', label='english')
        plt.plot(X_r[200:400, 0], X_r[200:400, 1], 'go', label='nihongo')
        plt.plot(X_r[400:600, 0], X_r[400:600, 1], 'bo', label='tieng viet')
        plt.legend(loc='upper right', shadow=False)

        # plt.subplot(212)
        # for color, i, target_name in zip(colors, [0, 1, 2], languages):
        #     plt.scatter(X_r2[y == i, 0], X_r2[y == i, 1], alpha=.8, color=color,
        #                 label=target_name)
        plt.figure()
        plt.title('LDA of dataset')
        lda = LinearDiscriminantAnalysis(n_components=2)
        X_r2 = lda.fit(X, y).transform(X)
        print(X_r2.shape)
        plt.plot(X_r2[0:200, 0], X_r2[0:200, 1], 'ro', label='english')
        plt.plot(X_r2[200:400, 0], X_r2[200:400, 1], 'go',label='nihongo')
        plt.plot(X_r2[400:600, 0], X_r2[400:600, 1], 'bo',label='tieng viet')
        plt.legend(loc='upper right', shadow=False)
        # plt.title('LDA of IRIS dataset')

        plt.show()

    def calcF0(self, signal):
        from python_speech_features import mfcc
        i = 0
        self.F0 = []
        N = self.frameLen
        count = 0
        while len(signal)%N != 0:
            signal.append(0)
        count1 =0
        count2 =200
        count3 =400
        for i in range(0, len(signal), N):
            self.F0.append(self.getF0OfFrame2(signal[i:i+N], N))
            if (self.choose ==1):
                self.audio1.append(self.getF0OfFrame2(signal[i:i+N], N))
                mfcc_signal = np.asanyarray(signal[i:i+N])
                mfcc_audio1 = mfcc(mfcc_signal,self.fs,winlen=N/self.fs,winstep=N/(self.fs*2),numcep=13,nfft=self.frameLen)
                # print(mfcc_audio1)
                self.inputPCA[count1][0] = self.getF0OfFrame2(signal[i:i+N], N)
                for i in range (1,14):
                    self.inputPCA[count1][i] = mfcc_audio1[0][i-1]
                self.target.append(0)
                count1+=1
            if (self.choose ==2):
                self.audio2.append(self.getF0OfFrame2(signal[i:i+N], N))
                mfcc_signal = np.asanyarray(signal[i:i + N])
                mfcc_audio2 = mfcc(mfcc_signal, self.fs, winlen=N / self.fs, winstep=N / (self.fs * 2), numcep=13,nfft=self.frameLen)
                self.inputPCA[count2][0] = self.getF0OfFrame2(signal[i:i + N], N)
                for i in range (1,14):
                    self.inputPCA[count2][i] = mfcc_audio2[0][i-1]
                self.target.append(1)
                count2 += 1
            if (self.choose ==3 ):
                self.audio3.append(self.getF0OfFrame2(signal[i:i+N], N))
                mfcc_signal = np.asanyarray(signal[i:i + N])
                mfcc_audio3 = mfcc(mfcc_signal, self.fs, winlen=N / self.fs, winstep=N / (self.fs * 2), numcep=13,nfft=self.frameLen)
                # print(mfcc_audio3)
                self.inputPCA[count3][0] = self.getF0OfFrame2(signal[i:i + N], N)
                for i in range (1,14):
                    self.inputPCA[count3][i] = mfcc_audio3[0][i-1]
                self.target.append(2)
                count3 += 1
        # print(self.inputPCA)
        print(len(self.target))
        print(self.target)
        per = 1
        start = 0
        end = len(self.F0)
        for i in range(0, len(self.F0)):
            if self.F0[i] < 80 or self.F0[i] > 400:
                if i > len(self.F0)/2:
                    end = i
#                    print('end =', end)
                    break
                per = 0
            else:
                if per == 0:
                    start = i
#                    print('start =', start)
                per = 1
        self.F0 = self.F0[start:end]
        # print(self.F0)
        print(self.audio1)
        print(len(self.audio1))
        print(self.audio2)
        print(len(self.audio2))
        print(self.audio3)
        print(len(self.audio3))
        # print(mfcc_audio1)
        self.plotF0(self.F0, start)

    def browseFile1(self):
        from tkinter.filedialog import askopenfilename
        Tk().withdraw()
        self.fileName = askopenfilename()
        self.pathLabel['text'] = self.fileName
        self.fs, self.ip = wavfile.read(self.fileName)
        self.ip = [x/1000 for x in self.ip]
        if not self.plotted:
            self.ipCanvas.get_tk_widget().pack(fill=BOTH, expand=True)            #Pack axes widget
        self.plotIp()
        self.plotted = True
        self.resetSylReader()
        self.choose = 1

    def browseFile2(self):
        from tkinter.filedialog import askopenfilename
        Tk().withdraw()
        self.fileName = askopenfilename()
        self.pathLabe2['text'] = self.fileName
        self.fs, self.ip = wavfile.read(self.fileName)
        self.ip = [x/1000 for x in self.ip]
        if not self.plotted:
            self.ipCanvas.get_tk_widget().pack(fill=BOTH, expand=True)            #Pack axes widget
        self.plotIp()
        self.plotted = True
        self.resetSylReader()
        self.choose = 2

    def browseFile3(self):
        from tkinter.filedialog import askopenfilename
        Tk().withdraw()
        self.fileName = askopenfilename()
        self.pathLabe3['text'] = self.fileName
        self.fs, self.ip = wavfile.read(self.fileName)
        self.ip = [x/1000 for x in self.ip]
        if not self.plotted:
            self.ipCanvas.get_tk_widget().pack(fill=BOTH, expand=True)            #Pack axes widget
        self.plotIp()
        self.plotted = True
        self.resetSylReader()
        self.choose = 3

    def enableWidget(self, widget):
        widget['state'] = 'normal'

    def sylMark(self):
        self.checkInputData('All')
        self.plotIp()
        
        self.resetSylReader()
        
        es = float(self.esEntry.get())
        self.frameLen = int(self.frmEntry.get())
        
        frameNum = int(np.floor(len(self.ip)/self.frameLen))+2
        ipTemp = np.zeros(frameNum*self.frameLen-len(self.ip), int)
        ipTemp = np.append(self.ip, ipTemp)
        i = 0
        signal = 0
        sylStart = 0
        sylEnd = 0
        while i <= ipTemp.size-self.frameLen:
            frame = ipTemp[i:i+self.frameLen]
            e = np.mean(np.square(frame))
            if e > es:
                if signal == 0:
                    sylStart = i
                signal = 1
            else:
                if signal == 1:
                    sylEnd = i-1
                    if self.isSyllable(sylStart, sylEnd):
                        self.markSyl(sylStart, sylEnd)
                        self.saveSyl(sylStart, sylEnd)
                        signal = 0
            i=i+int(self.frameLen/2)
        self.updateCanvas(self.ipCanvas)
        self.opMenuChoices = list(range(1, len(self.sylList) + 1))
        self.initSylUI()
        
    def splitSyl(self, value):             
        self.currSyl = value
        self.plotSyl(self.currSyl)        
        self.F0 = []
        self.count = 0
        self.calcF0(self.sylList[self.currSyl-1])

    def getF0OfFrame2(self, x, N):
        K = 250
        for t in x:
            if abs(t) < 3:
                t = 0
        r = np.zeros(K+1, dtype='float')
        for k in range(0, K+1):  
            sum = 0
            for n in range (0, N-k):
                sum = sum + x[n]*x[n+k]
            r[k] = sum

        vmax = 0                
        kmax = 1
        for k in range(1, K):
            if r[k] > r[k-1] and r[k] > r[k+1]:
                if vmax < r[k]:
                    vmax = r[k]
                    kmax = k

        F0 = self.fs/kmax
        return F0

    def saveSyl(self, start, end):
        self.sylList.append(self.ip[start:end+1])
        
    def checkInputData(self, key):
        if self.fileName == '':            
            self.noFileWarning()
            return False
        if key=='File':
            return True
        if (self.esEntry.get() == '') or (self.frmEntry.get() == '') or (self.timeEntry.get() == ''):
            self.noEntryWarning()
            return False        
        
    def isSyllable(self, start, end):
        sylMinTime = int(self.timeEntry.get())
        return (end-start)>(sylMinTime/1000*self.fs)
    
    def markSyl(self, start, end):
        startX = [start, start]
        endX = [end, end]
        y = [max(self.ip), min(self.ip)]
        self.ipAx.plot(startX, y, 'g')
        self.ipAx.plot(endX, y, 'r')
        
    def updateCanvas(self, canvas):
        canvas.draw() 
        
    def playIp(self):
        from playsound import playsound
        self.checkInputData('File')
        playsound(self.fileName)
        
    def playSyl(self):
        import sounddevice as sd
        sd.play(self.playedSyl, self.fs)
        
    def plotIp(self):
        self.ipAx.clear()                                                         #Clear axes             
        self.ipAx.plot(self.ip)
        self.ipAx.set_xlabel('Time (ms)')
        self.ipAx.set_ylabel('Amplitude')
        
        #Set xtick labels to time(s)
        xtick = self.ipAx.get_xticks()  
        self.ipAx.set_xticklabels((xtick/self.fs*1000).astype(int))
        self.updateCanvas(self.ipCanvas)
    
    def plotSyl(self, key):
        self.sylAx.clear()
        self.sylAx.plot(self.sylList[key-1])
        self.playedSyl = self.sylList[key-1]
        
        self.sylAx.set_xlabel('Time (ms)')
        xtick = self.sylAx.get_xticks()  
        self.sylAx.set_xticklabels((xtick/self.fs*1000).astype(int))
        
        self.updateCanvas(self.sylCanvas)
        
        if not self.sylPlotted:
            self.sylCanvas = FigureCanvasTkAgg(self.sylFig, self.sylAxesFrame)
            self.sylCanvas.get_tk_widget().pack(fill=BOTH, expand=True)
        self.sylPlotted = True
        #self.f0Ax.plot(self.sylList[key+1])
        #self.f0Canvas.get_tk_widget().pack(fill=BOTH, expand=True)
        
    def plotFrame(self, frame):
        self.sylAx.clear()
        self.sylAx.plot(frame)
        self.playedSyl = frame
        self.sylAx.set_xlabel('Time (ms)')
        xtick = self.sylAx.get_xticks()  
        self.sylAx.set_xticklabels((xtick/self.fs*1000).astype(int))
        
        self.updateCanvas(self.sylCanvas)
        
        if not self.sylPlotted:
            self.sylCanvas = FigureCanvasTkAgg(self.sylFig, self.sylAxesFrame)
            self.sylCanvas.get_tk_widget().pack(fill=BOTH, expand=True)
        self.sylPlotted = True
        
    def plotF0(self, F0, start):
        self.f0Ax.clear()
        self.f0Ax.plot(F0, 'ro-')
        self.f0Ax.set_xlabel('Time (ms)')
        self.f0Ax.set_ylabel('Frequency (Hz)')
        xtick = self.f0Ax.get_xticks()
        self.f0Ax.set_xticklabels(((xtick + start)*self.frameLen/self.fs*1000).astype(int))
       # self.f0Axs.set_xlim([xmin,xmax])
        self.f0Ax.set_ylim([0, 800])
        self.updateCanvas(self.f0Canvas)
        
        if not self.f0Plotted:
            self.f0Canvas = FigureCanvasTkAgg(self.f0Fig, self.f0AxesFrame)
            self.f0Canvas.get_tk_widget().pack(fill=BOTH, expand=True)
        self.f0Plotted = True
        
    def resetSylReader(self):
        if self.sylFnFrame != None:
            self.sylFnFrame.destroy()
        if self.sylPlotted:
            self.sylCanvas.get_tk_widget().destroy()
            self.sylPlotted = False
        if self.f0Plotted:
            self.f0Canvas.get_tk_widget().destroy()
            self.f0Plotted = False
        self.sylList = []
        self.opMenuVar.set('')
        
    def noFileWarning(self):
        mbox.showwarning('Thông báo', 'Chưa chọn file')
        
    def noEntryWarning(self):
        mbox.showwarning('Thông báo', 'Chưa nhập thông số')
        
root = Tk()
root.state('zoomed')
window = Window(root)
#root.attributes("-fullscreen",True)
root.mainloop()