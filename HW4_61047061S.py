from tkinter import *
from tkinter import filedialog # to pop open file dialog
from tkinter import ttk
import cv2 as cv # 'opencv' module to handle image 
from PIL import ImageTk, Image # 'pillow' module to handle image
import math
import matplotlib.pyplot as plt # help us draw the diagram
# the following two help us to embed pyplot chart into tkinter UI
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
from random import random, uniform
import numpy as np

# Window grid configuration:
# ----------------------------
# input_f | funct_f | output_f
# ----------------------------
#         analysis_f
# ----------------------------

# tkinter building concept: build the widgets, and then put then on screen.
# build a frame for window, which is the toplevel window that will contain everything else
root = Tk()
# Tcl manages widgets by its paths. we can use print(str(widget)) to check.
#print(str(root))
root.title("AIP 61047061S") # title name
STD:float = None

'''
In the future, hope to make a zoom-in/out function with mouse' wheel.
Here, I just modify the size with easy math, and not work with too tiny images.
'''

'''------------------Other Function------------------'''
def Init_window(): # initialize UI
    global STD
    global Haar_Lv
    # init frame size
    frame_width, frame_height = 1, 1 # notice that unit are NOT pixels
    input_frame.configure(width=frame_width, height=frame_height)
    output_frame.configure(width=frame_width, height=frame_height)
    chart_frame.configure(height=frame_height)
    #print("frame resize completed.")

    # init parameters
    STD = None
    Haar_Lv = None

    # init preview image
    input_canvas.configure(image="")
    output_canvas.configure(image="")

    button_list = funct_frame.winfo_children()
    # button_list[0] = open file button, last two are 'clear' and 'exit', which no need to disable them.
    for i in range(1,len(button_list)-2): 
        button_list[i].configure(state=DISABLED)
    #print("button state init completed")

    # loop through the chart_frame and then delete charts.
    # p.s. .winfo_children return a list of children widgets.
    if len(chart_frame.winfo_children())!=0:
        for charts in chart_frame.winfo_children():
            charts.destroy()

    #print("charts init completed")
def Resize_img(image): # image type = np.array
    # convert read file from nparray to tk-image and resize or previewing, avoiding too large display
    # PIL: image.resize((x-size, y-size), Image.ANTIALIAS: for not ruining the quality of the pic)
    scale_percent = math.ceil(100/(max(image.shape)/800)) # percent of original size
    #print("Scaling: ", scale_percent)
    # .shape return pixels in unit so is opencv's resize methods
    if max(image.shape) >= 800:   
        width = int(math.ceil(image.shape[1] * scale_percent / 100))
        height = int(math.ceil(image.shape[0] * scale_percent / 100))
        resized_img_dim = (width, height)
        resized_img = cv.resize(image, resized_img_dim)
    else:
        resized_img = image

    if image.ndim == 3: # colorful image, ndim: attribute of a np.array which is dimension.
        # rearrange the BGR to typical RGB
        blue, green, red = cv.split(resized_img)
        img_display = cv.merge((red,green,blue))
        img_display = Image.fromarray(img_display)
    else: # gray-level image which no need to rearrange BGR to RGB
        img_display = Image.fromarray(resized_img)
    # transform to the widget that tkinter can use.
    img_display = ImageTk.PhotoImage(img_display) # for display on screen
    return img_display

'''------------------Open/Save File Functions------------------'''
def Open_a_file(): # read a file assigned by the user
    global input_img
    global input_img_display
    global gray_img
    global frame_width
    global frame_height
    
    # pop an open file window, and we can restrict file types,
    # and it only gets the dir of the user's file location.
    input_img_path = filedialog.askopenfilename(initialdir="C:\\Desktop", title="Select an image",
                                                filetypes=(("JPG files","*.jpg"),
                                                           ("PNG files","*.png"),
                                                           ("PPM files","*.ppm"),
                                                           ("BMP files","*.bmp") )
                                               )
    if input_img_path: # to check if the dir is empty
        Init_window() # initialize some widgets and window sizes.
        # create a img object and display via canvas
        input_img = cv.imread(input_img_path) # read original file for processing
        gray_img = cv.cvtColor(input_img, cv.COLOR_BGR2GRAY) # convert image to gray scale
        input_img_display = Resize_img(input_img)
        input_canvas.configure(image=input_img_display) # adjust input label to show it
        output_canvas.configure(image=input_img_display) # show when load an img even tho it's not modified yet
        #print("Operation Ends.")

        # Enable functional Buttons
        do_sth_button.configure(state=NORMAL)
        histogram_button.configure(state=NORMAL)
        GWN_button.configure(state=NORMAL)
        Haar_button.configure(state=NORMAL)
        save_file_button.configure(state=NORMAL)

        return input_img
def Save_file(output_image):
    # get output image save path as str
    output_img_path = filedialog.asksaveasfilename(title="Save as BMP file.",filetypes=[("All files","*")],defaultextension=".bmp")
    if output_img_path: # check if the dir is empty
        #print(output_img_path)
        # convert image to .bmp file and save image to some directory
        cv.imwrite(f"{output_img_path}", output_image)


def Do_something(input_image): 
    global output_img
    global output_img_display
    pass # do some processing or through some funct with the next line below.
    output_img = Some_processing(input_image) # original file after some processing
    output_img_display = Resize_img(output_img) # for display on screen
    output_canvas.configure(image=output_img_display) # adjust output label to show img
    save_file_button.configure(state=NORMAL) # Enable the do-sth button
def Some_processing(input_image): # TBD
    output_img = input_image
    return output_img

'''------------------Histogram Functions------------------'''
def BGR_Histo(image): # input image is read by opencv which is BGR array.
    global bgr_histo_canvas
    COLOR = ('b','g','r')
    bgr_histo_fig = Figure(figsize=(6.5,4), dpi=100) # create a figure with size 600*400
    bgr_histo_subplot = bgr_histo_fig.add_subplot(title="RGB Histogram",
                                                  xlabel="Scale",
                                                  ylabel="Accumulation (# of pixels)")
    for i, color in enumerate(COLOR):
        bgr_histo_arr = cv.calcHist([image], [i], None, [256], [0, 256]) # channel:(0,1,2) = (b,g,r)
        #bgr_histo_arr = bgr_histo_arr.ravel()
        bgr_histo_subplot.plot((range(0,256)), bgr_histo_arr, color=color)

    # set histogram's parent to chart_fram.
    bgr_histo_canvas = FigureCanvasTkAgg(bgr_histo_fig, master=chart_frame)  # A tk.DrawingArea.
    bgr_histo_canvas.draw()
    # toolbar itself is also a widget
    bgr_histo_fig_toolbar = NavigationToolbar2Tk(bgr_histo_canvas, chart_frame, pack_toolbar=False)
    bgr_histo_fig_toolbar.update()
    bgr_histo_fig_toolbar.grid(row=0,column=0)
    bgr_histo_canvas.get_tk_widget().grid(row=1, column=0)
    return bgr_histo_canvas
def Gray_Histo(gray_img, title_name:str): # draw gray scale histogram at right-bottom of the root win.
    global histo_canvas
    # calculate the histogram of gray level.
    # calcHist([image], [#channel], mask, [#bin], [range of pixel value])
    histo_arr = cv.calcHist([gray_img], [0], None, [256], [0, 256]) # this returns a 1*256 array
    histo_arr = histo_arr.ravel() # reshape to 1D array, or use histogram.reshape(-1)
    histo_fig = Figure(figsize=(6.5,4), dpi=100) # create a figure with size 600*400
    histo_subplot = histo_fig.add_subplot(title=title_name,
                                          xlabel="Grayscale Value",
                                          ylabel="Accumulation (# of pixels)")
    # draw the histogram according to the array calculated by opencv.calchist
    histo_subplot.bar((range(0,256)), histo_arr, width=1.0, color="GRAY", edgecolor="BLACK")
    if STD: # add standard deviation text on the top-left corner in histogram
        histo_subplot.text(1, 1, f'σ = {STD}',
                            verticalalignment='top', horizontalalignment='right',
                            transform=histo_subplot.transAxes,
                            color='blue', fontsize=10)

    # temporarily, I set historgram's parent to root.
    histo_canvas = FigureCanvasTkAgg(histo_fig, master=chart_frame)  # A tk.DrawingArea.
    histo_canvas.draw()
    # toolbar itself is also a widget
    histo_fig_toolbar = NavigationToolbar2Tk(histo_canvas, chart_frame, pack_toolbar=False)
    histo_fig_toolbar.update()
    histo_fig_toolbar.grid(row=0,column=1)
    histo_canvas.get_tk_widget().grid(row=1, column=1)
def Histogram(image): # generate a histogram of the input image
    global output_img_display
    global output_img
    # gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY) # convert image to gray scale
    output_img_display = Resize_img(gray_img)
    output_canvas.configure(image=output_img_display) # output preview image becomes gray-level
    # chart result
    Gray_Histo(gray_img,"Gray Level Histogram") # gray_img is generate right after opening the file
    BGR_Histo(image) # input_img as BGR read by opencv.imread()

    output_img = gray_img
    # Enable the buttons
    save_file_button.configure(state=NORMAL) # Enable save-as button
    
'''------------------Gaussian White Noise Functions------------------'''
def GWN_gen(STD, gray_img): # this function returns a np.array of added noise image array.
    global output_img
    GWN = []
    noise_img = []
    for row in range(len(gray_img)):
        GWN.append([])
        noise_img.append([])
        for col in range(0,len(gray_img[0]),2): # two pair going through
            '''
            random.random() generates float, x = [0,1)
            adjust to 1-x = (0,1] for the natural log calculation.
            random.unirom(a,b) generates a <= float <= b
            '''
            R, PHI = 1-random(), uniform(0,1)
            #print(R, PHI)
            # Box-Muller Transform
            z1 = STD*math.cos(2*math.pi*PHI)*(math.sqrt(-2*math.log(R)))
            z2 = STD*math.sin(2*math.pi*PHI)*(math.sqrt(-2*math.log(R)))
            #print(z1,z2)
            if col+1 >= len(gray_img[0]): # last col
                # f stands for modified image pixel
                f = gray_img[row][col] + round((z1+z2)/2)
                #print("f: ", f)
                if f < 0:
                    f = 0
                if f > 255:
                    f = 255
                noise_img[row].append(f)
                GWN[row].append((z1+z2)/2)
            else:
                GWN[row].append(z1)
                GWN[row].append(z2)
                f1 = gray_img[row][col] + round(z1)
                f2 = gray_img[row][col+1] + round(z2)
                #print("f1, f2 = ",(f1,f2))
                if f1 < 0:
                    f1 = 0
                if f1 > 255:
                    f1 = 255
                if f2 < 0:
                    f2 = 0
                if f2 > 255:
                    f2 = 255
                noise_img[row].append(f1)
                noise_img[row].append(f2)
    # transform list to np.array
    noise_img = np.array(noise_img, dtype="uint8") # we have to assign the data type
    output_img = noise_img
    GWN_arr = np.array(GWN, dtype="float32") # we have to assign the data type

    '''adding additional graph for noises chart, not normalized'''
    GWN_x_ax = []
    GWN_y_ax = []
    for x,y in enumerate(GWN_arr):
        GWN_x_ax.append(x)
        GWN_y_ax.append(y)
    GWN_fig = Figure(figsize=(6.5,4), dpi=100) # create a figure with size 600*400
    GWN_subplot = GWN_fig.add_subplot(title="Noises",
                                      xlabel="Generated Order",
                                      ylabel="Value (float)")
    GWN_subplot.plot(GWN_x_ax, GWN_y_ax, color='gray')
    GWN_subplot.set_xlim(0,255) # set x-axis two edges to be 0 and 255, no extra space.
    # add standard deviation text in the graph.
    GWN_subplot.text(1, 1, f"σ = {STD}",
                            verticalalignment='top', horizontalalignment='right',
                            transform=GWN_subplot.transAxes,
                            color='blue', fontsize=10)
    GWN_canvas = FigureCanvasTkAgg(GWN_fig, master=chart_frame)  # A tk.DrawingArea.
    GWN_canvas.draw()
    # toolbar itself is also a widget
    GWN_fig_toolbar = NavigationToolbar2Tk(GWN_canvas, chart_frame, pack_toolbar=False)
    GWN_fig_toolbar.update()
    GWN_fig_toolbar.grid(row=0,column=0)
    GWN_canvas.get_tk_widget().grid(row=1, column=0)
    '''
    normalized GWN array to image range of [0,255]
    np.ptp(arr, axis=None, out=None, keepdims=<no value>) >>> Range of values (max-min) along an axis.
    min-max scaling: https://en.wikipedia.org/wiki/Feature_scaling#Rescaling_(min-max_normalization)
    '''
    GWN_norm_arr = (255*(GWN_arr - np.min(GWN_arr))/np.ptp(GWN_arr)).astype(np.uint8) # shape = gray_image
    return noise_img, GWN_norm_arr # np.array of the image added GWN result, that is, img + noise
def Confirm_STD_and_Draw(STD_input:str, gray_img):
    global STD_chk_label
    global STD
    global output_img_display
    # convert to float or integer(.0)
    try:
        float(STD_input)
        if float(STD_input) < 0:
            STD_chk_label.configure(text="Invalid Value of Standard Deviation.", background="yellow")
        else:
            STD = float(STD_input)
            #print("User input STD = ", STD)
            #print("Standard Deviation: ", STD)
            noised_img, z1_z2_distri = GWN_gen(STD, gray_img) # output:(noise_img, GWN_norm)

            # output display the result of adding noises.
            output_img_display = Resize_img(noised_img) 
            output_canvas.configure(image=output_img_display)

            # draw the z1, z2 (noises adder) to see if it's Gaussian Distribution.
            Gray_Histo(z1_z2_distri, "Gaussian White Noises Histogram (Normalized)")
            STD_ask_win.destroy() # close the standard deviation input window.
            # Enable the buttons
            save_file_button.configure(state=NORMAL) # Enable save-as button
    except ValueError:    
        STD_chk_label.configure(text="Invalid Input. Please Enter a number.", background="yellow")
        STD_entry.delete(0,END)
def GWN(gray_img):
    '''establish the ask window for user to input the standard deviation of noises.'''
    global STD_ask_win
    global STD_chk_label
    global STD_entry
    # pop-out window to ask the STD value for user to enter.
    STD_ask_win = Toplevel(root)
    #STD_ask_win.geometry("300x200")
    STD_ask_win.title("Standard Deviation Setting.")
    # adjust the pop window location
    win_x,win_y = funct_frame.winfo_rootx(), root.winfo_rooty()+200
    STD_ask_win.geometry(f'+{win_x}+{win_y}')
    STD_input_label = ttk.Label(STD_ask_win, text="Please Enter the Standard Deviation of Gaussian White Noise:")
    STD_entry = ttk.Entry(STD_ask_win)
    win_button_frame = ttk.Frame(STD_ask_win)
    STD_confirm_button = ttk.Button(win_button_frame, text="Confirm",
                                    command=lambda:Confirm_STD_and_Draw(STD_entry.get(), gray_img))
    STD_clear_button = ttk.Button(win_button_frame, text="Clear",
                                  command=lambda:STD_entry.delete(0,END)) # clear entry
    STD_exit_button = ttk.Button(win_button_frame, text="Cancel",
                                 command=STD_ask_win.destroy)
    STD_chk_label = ttk.Label(STD_ask_win, text="")

    STD_input_label.grid(row=0,column=0)
    STD_entry.grid(row=1,column=0, columnspan=3)
    win_button_frame.grid(row=2,column=0)
    STD_confirm_button.grid(row=0,column=0)
    STD_clear_button.grid(row=0,column=1)
    STD_exit_button.grid(row=0,column=2)

    STD_chk_label.grid(row=3,column=0)

'''------------------Discrete Wavelet (Haar function)----------------'''
def Ask_Haar_Lv():
    '''establish the ask window for user to input DWT level.'''
    global Haar_Lv_ask_win
    global Haar_Lv_chk_label
    global Haar_Lv_entry
    # pop-out window to ask the Haar_Lv value for user to enter.
    Haar_Lv_ask_win = Toplevel(root)
    #Haar_Lv_ask_win.geometry("300x200")
    Haar_Lv_ask_win.title("Haar DWT level Setting.")
    # adjust the pop window location
    win_x,win_y = funct_frame.winfo_rootx(), root.winfo_rooty()+200
    Haar_Lv_ask_win.geometry(f'+{win_x}+{win_y}')
    Haar_Lv_input_label = ttk.Label(Haar_Lv_ask_win, text="Please Enter the level of Haar DWT:")
    Haar_Lv_entry = ttk.Entry(Haar_Lv_ask_win) 
    win_button_frame = ttk.Frame(Haar_Lv_ask_win)
    Haar_Lv_confirm_button = ttk.Button(win_button_frame, text="Confirm",
                                    command=lambda:Confirm_Lv(Haar_Lv_entry.get()))
    Haar_Lv_clear_button = ttk.Button(win_button_frame, text="Clear",
                                  command=lambda:Haar_Lv_entry.delete(0,END)) # clear entry
    Haar_Lv_exit_button = ttk.Button(win_button_frame, text="Cancel",
                                 command=Haar_Lv_ask_win.destroy)
    Haar_Lv_chk_label = ttk.Label(Haar_Lv_ask_win, text="")

    Haar_Lv_input_label.grid(row=0,column=0)
    Haar_Lv_entry.grid(row=1,column=0, columnspan=3)
    win_button_frame.grid(row=2,column=0)
    Haar_Lv_confirm_button.grid(row=0,column=0)
    Haar_Lv_clear_button.grid(row=0,column=1)
    Haar_Lv_exit_button.grid(row=0,column=2)

    Haar_Lv_chk_label.grid(row=3,column=0)
def Confirm_Lv(Haar_Lv_input:str):
    MAX_LV = 8
    global Haar_Lv_chk_label
    global Haar_Lv
    global output_img_display
    global output_img
    # convert to float or integer(.0)
    try:
        int(Haar_Lv_input)
        if 0 < int(Haar_Lv_input) <= MAX_LV:
            Haar_Lv = int(Haar_Lv_input)
            #print("User input Haar_Lv = ", Haar_Lv)
            #print("Standard Deviation: ", Haar_Lv)

            Haar_Lv_ask_win.destroy() # close the standard deviation input window.

            # output display the result of 2D-DWT.
            output_img = Haar(gray_img, Haar_Lv)
            output_img_display = Resize_img(output_img) # for display on screen
            output_canvas.configure(image=output_img_display)

            # Enable the buttons
            save_file_button.configure(state=NORMAL) # Enable save-as button

        else:
            Haar_Lv_chk_label.configure(text="Out of level range (MAX=8).", background="yellow")

    except ValueError:    
        Haar_Lv_chk_label.configure(text="Invalid Input. Please Enter a number.", background="yellow")
        Haar_Lv_entry.delete(0,END)
def Haar(gray_img, lv=2):
    res_by_lv = [[]]*lv
    # resize image to the size of power of 2 > take 512 here
    resized_gray_img = cv.resize(gray_img, (512, 512), interpolation=cv.INTER_NEAREST)
    iter_list = Haar_DWT_Cal(resized_gray_img) # return DWT[0~3] type:list
    res_by_lv[lv-1] = iter_list
    '''
    DWT[0] | DWT[1]
    ---------------
    DWT[2] | DWT[3]
    '''
    # save every level result
    while lv-1 > 0:
        iter_list = Haar_DWT_Cal(iter_list[0])
        res_by_lv[lv-2] = iter_list
        lv -= 1
    # contactate
    for i,lv_res in enumerate(res_by_lv):
        #print(i,lv_res)
        if i == 0:
            con_row1 = np.concatenate((lv_res[0],lv_res[1]), axis=1)
            con_row2 = np.concatenate((lv_res[2],lv_res[3]), axis=1)
            res = np.concatenate((con_row1,con_row2), axis=0)
        else:
            con_row1 = np.concatenate((res,lv_res[1]), axis=1)
            con_row2 = np.concatenate((lv_res[2],lv_res[3]), axis=1)
            res = np.concatenate((con_row1,con_row2), axis=0)
    # print(type(res))
    # print(res.shape)
    return res
def Haar_DWT_Cal(arr):
    if arr.shape[0] == 1 or arr.shape[1] == 1:
        return arr
    else:
        arr = arr.astype(np.uint16)
        res = [[],[],[],[]]
        row, col = arr.shape[0], arr.shape[1]
        # print(row,col)
        '''
        calculation result:
        res11 | res12
        ------|-------
        res21 | res22
        '''
        for i in range(0, row, 2):
            for j in range(0, col, 2):
                # since the img array dtype = uint8, be careful of the overflow
                A,B=arr[i][j],arr[i][j+1]
                C,D=arr[i+1][j],arr[i+1][j+1]
                LL = round((A+B+C+D)*0.25)
                HL = round((A-B+C-D)*0.25)
                LH = round((A+B-C-D)*0.25)
                HH = round((A-B-C+D)*0.25)
                if LL < 0:
                    LL = 0
                if HL < 0:
                    HL = 0
                if LH < 0:
                    LH = 0
                if HH < 0:
                    HH = 0
                res[0].append(LL) # res11
                res[1].append(HL) # res12
                res[2].append(LH) # res21
                res[3].append(HH) # res22
        for i in range(4):
            res[i] = np.array(res[i]).astype(np.uint8) # type:np.array while res.type=list
            res[i] = np.array(res[i]).reshape(row//2,col//2) # type:np.array while res.type=list
            #print("res[i] array: ",res[i])
        return res # array list
                


# create main content frame
input_frame = ttk.LabelFrame(root, text="Input Image", borderwidth=5, relief="ridge")
input_frame.grid(row=0, column=0, sticky=(N,S,E,W))

funct_frame = ttk.Frame(root, padding=5)
funct_frame.grid(row=0, column=1, sticky=(N))

output_frame = ttk.LabelFrame(root, text="Output Preview", borderwidth=5, relief="ridge")
output_frame.grid(row=0, column=2, sticky=(N,S,E,W))

chart_frame = ttk.LabelFrame(root, height=40, text="Analysis", borderwidth=5, relief="ridge")
chart_frame.grid(row=1, column=0, sticky=(N,S,E,W), columnspan=3)

# these config below tells frame should expand to fill any extra space if the window is resized.
# (widget).columnconfigure(col_index, weight) # weight = ratio stretch when dragging
# (widget).rowconfigure(row_index, weight)
# input_frame
root.columnconfigure(0, weight=1)
root.rowconfigure(0, weight=1)
#output_frame
root.columnconfigure(2, weight=1)
root.rowconfigure(0, weight=1)
# chart_frame
root.columnconfigure(0, weight=1)
root.rowconfigure(1, weight=1)
# add extra-space to let it averagely be placed.
chart_frame.columnconfigure((0,1), weight=1)
input_frame.columnconfigure(0, weight=1)
output_frame.columnconfigure(0, weight=1)

# create a button obj: where it puts, what text on it, and so on.
# command = (funct.name) or you can use lambda if we need to pass a parameter.
open_file_button = ttk.Button(funct_frame, text="Open A File",
                              command=Open_a_file)
open_file_button.grid(row=0, column=1, sticky=(N))
do_sth_button = ttk.Button(funct_frame, text="Do Sth.",
                           command=lambda:Do_something(input_img), state=DISABLED)
do_sth_button.grid(row=1, column=1, sticky=(N))
histogram_button = ttk.Button(funct_frame, text="Histogram",
                              command=lambda:Histogram(input_img), state=DISABLED)
histogram_button.grid(row=2, column=1, sticky=(N))
GWN_button = ttk.Button(funct_frame, text="Guassian",
                              command=lambda:GWN(gray_img), state=DISABLED)
GWN_button.grid(row=3, column=1, sticky=(N))
Haar_button = ttk.Button(funct_frame, text="Haar",
                              command=Ask_Haar_Lv, state=DISABLED)
Haar_button.grid(row=4, column=1, sticky=(N))
save_file_button = ttk.Button(funct_frame, text="Save As",
                              command=lambda:Save_file(output_img), state=DISABLED)
save_file_button.grid(row=5, column=1, sticky=(N))
clear_button = ttk.Button(funct_frame, text="Clear", command=Init_window)
clear_button.grid(row=6, column=1, sticky=(N))
quit_button = ttk.Button(funct_frame, text="Exit", command=root.quit)
quit_button.grid(row=7, column=1, sticky=(N))


# display input & output image via Label
input_canvas = ttk.Label(input_frame)
input_canvas.grid()
output_canvas = ttk.Label(output_frame)
output_canvas.grid()

#root.iconbitmap("(file address)") # the icon of the window
root.mainloop() # event loop
