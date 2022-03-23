from tkinter import *
from tkinter import filedialog # to pop open file dialog
from tkinter import ttk
from cv2 import imread, imwrite, resize, split, merge
#import cv2 as cv # 'opencv' module to handle image 
from PIL import ImageTk, Image # 'pillow' module to handle image
# tkinter building concept: build the widgets, and then put then on screen.
# build a frame for window, which is the toplevel window that will contain everything else
root = Tk()
# Tcl manages widgets by its paths. we can use print(str(widget)) to check.
#print(str(root))
root.title("AIP 61047061S") # title name

def Resize_img(image): 
    # convert read file from nparray to tk-image and resize or previewing, avoiding too large display
    # PIL: image.resize((x-size, y-size), Image.ANTIALIAS: for not ruining the quality of the pic)
    scale_percent = 100 # percent of original size
    width = int(round(image.shape[1] * scale_percent / 100))
    height = int(round(image.shape[0] * scale_percent / 100))
    dim = (width, height)
    resized_img = resize(image, dim)
    # rearrange the BGR to typical RGB
    blue, green, red = split(resized_img)
    img_display = merge((red,green,blue))
    img_display = Image.fromarray(img_display)
    # transform to the widget that tkinter can use.
    img_display = ImageTk.PhotoImage(img_display) # for display on screen
    return img_display

def Open_a_file(): # read a file assigned by the user
    global input_img
    global input_img_display
    global frame_width
    global frame_height
    frame_width, frame_height = 40,40
    # pop an open file window, and we can restrict file types,
    # and it only gets the dir of the user's file location.
    input_img_path = filedialog.askopenfilename(initialdir="C:\\Desktop", title="Select an image",
                                                filetypes=(("JPG files","*.jpg"),
                                                           ("PNG files","*.png"),
                                                           ("PPM files","*.ppm"),
                                                           ("BMP files","*.bmp") )
                                               )
    if input_img_path: # to check if the dir is empty
        # init
        save_file_button.configure(state=DISABLED)
        # init input and output preview frame size
        input_frame.configure(width=frame_width, height=frame_height)
        output_frame.configure(width=frame_width, height=frame_height)
        # create a img object and display via canvas
        input_img = imread(input_img_path) # read original file for processing
        input_img_display = Resize_img(input_img)
        input_canvas.configure(image=input_img_display) # adjust input label to show it
        output_canvas.configure(image=input_img_display) # show when load an img even tho it's not modified yet
        #print("Operation Ends.")
        do_sth_button.configure(state=NORMAL) # Enable the do-sth button
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

def Save_file(image):
    # get output image save path as str
    output_img_path = filedialog.asksaveasfilename(title="Save as BMP file.",filetypes=[("All files","*")],defaultextension=".bmp")
    if output_img_path: # check if the dir is empty
        #print(output_img_path)
        # convert image to .bmp file and save image to some directory
        imwrite(f"{output_img_path}", image)

# create main content frame
funct_frame = ttk.Frame(root, padding=5)
funct_frame.grid(row=0, column=1, sticky=(N))
# these config below tells frame should expand to fill any extra space if the window is resized.
# (widget).columnconfigure(col_index, weight) # weight = ratio stretch when dragging
# (widget).rowconfigure(row_index, weight)
root.columnconfigure(0, weight=1)
root.rowconfigure(0, weight=1)
root.columnconfigure(2, weight=1)
root.rowconfigure(0, weight=1)

input_frame = ttk.LabelFrame(root, text="Input Image", borderwidth=5, relief="ridge")
input_frame.grid(row=0, column=0, sticky=(N,S,E,W))
output_frame = ttk.LabelFrame(root, text="Output Preview", borderwidth=5, relief="ridge")
output_frame.grid(row=0, column=2, sticky=(N,S,E,W))

# create a button obj: where it puts, what text on it, and so on.
# command = (funct.name) or you can use lambda if we need to pass a parameter.
open_file_button = ttk.Button(funct_frame, text="Open A File",
                              command=Open_a_file)
open_file_button.grid(row=0, column=1, sticky=(N))

do_sth_button = ttk.Button(funct_frame, text="Do Sth.",
                           command=lambda:Do_something(input_img), state=DISABLED)
do_sth_button.grid(row=1, column=1, sticky=(N))

save_file_button = ttk.Button(funct_frame, text="Save As",
                              command=lambda:Save_file(output_img), state=DISABLED)
save_file_button.grid(row=2, column=1, sticky=(N))

quit_button = ttk.Button(funct_frame, text="Exit", command=root.quit)
quit_button.grid(row=3, column=1, sticky=(N))


# display input & output image via Label
input_canvas = ttk.Label(input_frame, width=4)
input_canvas.grid()
output_canvas = ttk.Label(output_frame, width=4)
output_canvas.grid()

#root.iconbitmap("(file address)") # the icon of the window
root.mainloop() # event loop
