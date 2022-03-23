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

def Init_window():
    # init frame size
    frame_width, frame_height = 40, 40 # notice that unit are NOT pixels
    input_frame.configure(width=frame_width, height=frame_height)
    output_frame.configure(width=frame_width, height=frame_height)
    chart_frame.configure(height=frame_height)
    #print("frame resize completed.")

    # init preview image
    input_canvas.configure(image="")
    output_canvas.configure(image="")

    # Enable the buttons
    do_sth_button.configure(state=NORMAL) # Enable the do-sth button
    histogram_button.configure(state=NORMAL) # Enable the Histrogram button
    #print("button state init completed")

    # loop through the chart_frame and then delete charts.
    # p.s. .winfo_children return a list of children widgets.
    if len(chart_frame.winfo_children())!=0:
        for charts in chart_frame.winfo_children():
            charts.destroy()

    #print("charts init completed")

'''
In the future, hope to make a zoom-in/out function with mouse' wheel.
Here, I just modify the size with easy math, and not work with too tiny images.
'''
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

def Open_a_file(): # read a file assigned by the user
    global input_img
    global input_img_display
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
        input_img_display = Resize_img(input_img)
        input_canvas.configure(image=input_img_display) # adjust input label to show it
        output_canvas.configure(image=input_img_display) # show when load an img even tho it's not modified yet
        #print("Operation Ends.")
        return input_img

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

def Gray_Histo(image):
    global histo_canvas
    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY) # convert to gray scale
    # calculate the histogram of gray level.
    # calcHist([image], [#channel], mask, [#bin], [range of pixel value])
    histo_arr = cv.calcHist([image], [0], None, [256], [0, 256]) # this returns a 1*256 array
    histo_arr = histo_arr.ravel() # reshape to 1D array, or use histogram.reshape(-1)
    histo_fig = Figure(figsize=(6.5,4), dpi=100) # create a figure with size 600*400
    histo_subplot = histo_fig.add_subplot(title="Gray Level Histogram",
                                          xlabel="Grayscale Value",
                                          ylabel="Accumulation (# of pixels)")
    histo_subplot.bar((range(0,256)), histo_arr, width=1.0, color="GRAY", edgecolor="BLACK")

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
    gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY) # convert image to gray scale
    output_img_display = Resize_img(gray_image)
    output_canvas.configure(image=output_img_display) # output preview image becomes gray-level
    # chart result
    Gray_Histo(image)
    BGR_Histo(image)

    output_img = gray_image
    # Enable the buttons
    save_file_button.configure(state=NORMAL) # Enable save-as button
    
def Save_file(image):
    # get output image save path as str
    output_img_path = filedialog.asksaveasfilename(title="Save as BMP file.",filetypes=[("All files","*")],defaultextension=".bmp")
    if output_img_path: # check if the dir is empty
        #print(output_img_path)
        # convert image to .bmp file and save image to some directory
        cv.imwrite(f"{output_img_path}", image)

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
chart_frame.columnconfigure((0,1), weight=1)

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

save_file_button = ttk.Button(funct_frame, text="Save As",
                              command=lambda:Save_file(output_img), state=DISABLED)
save_file_button.grid(row=3, column=1, sticky=(N))

quit_button = ttk.Button(funct_frame, text="Exit", command=root.quit)
quit_button.grid(row=4, column=1, sticky=(N))


# display input & output image via Label
input_canvas = ttk.Label(input_frame)
input_canvas.grid()
output_canvas = ttk.Label(output_frame)
output_canvas.grid()

#root.iconbitmap("(file address)") # the icon of the window
root.mainloop() # event loop
