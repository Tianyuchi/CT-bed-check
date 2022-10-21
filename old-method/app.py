import os, sys
import random
import time

import tkinter as tk
import tkinter.filedialog
import tkinter.messagebox
from tkinter import ttk

from dict_learn import DetectionAsDict
from utils_io import TopBed

class App(object):
    def __init__(self, master):
        super().__init__()
        self.master = master
        self.frm = ttk.Frame(master)
        #self.frm['borderwidth'] = 5
        self.frm.grid(column=0, row=0, sticky=(tk.N, tk.S, tk.E, tk.W))
        self.frm.columnconfigure(1, weight=1)
        self.frm.rowconfigure(0, weight=1)
        self.frm.rowconfigure(1, weight=1)

        self.btn_DICOM = ttk.Button(self.frm, text="DICOM Folder:", command=self.btn_DICOM_act)
        self.btn_DICOM.grid(column=0, row=0, sticky=(tk.N, tk.S, tk.E, tk.W), pady=5, padx=2)

        self.btn_output = ttk.Button(self.frm, text="Output Folder:", command=self.btn_output_act)
        self.btn_output.grid(column=0, row=1, sticky=(tk.N, tk.S, tk.E, tk.W), pady=5, padx=2)

        self.entry_DICOM = ttk.Entry(self.frm, text=os.getcwd())
        self.entry_DICOM.grid(column=1, row=0, sticky=(tk.N, tk.S, tk.E, tk.W), pady=5, padx=2)
        self.entry_DICOM_content = tk.StringVar()
        self.entry_DICOM["textvariable"] = self.entry_DICOM_content

        self.entry_output = ttk.Entry(self.frm, text=os.getcwd())
        self.entry_output.grid(column=1, row=1, sticky=(tk.N, tk.S, tk.E, tk.W), pady=5, padx=2)
        self.entry_output_content = tk.StringVar()
        self.entry_output["textvariable"] = self.entry_output_content

        self.label_status = ttk.Label(self.frm, text="Ready")
        self.label_status.grid(column=2, row=0, sticky=(tk.N, tk.S, tk.E, tk.W), pady=5, padx=5)

        self.btn_run = ttk.Button(self.frm, text="Run", command=self.btn_run_act)
        self.btn_run.grid(column=2, row=1, sticky=(tk.N, tk.S, tk.E, tk.W), pady=5, padx=5)
        #ttk.Button(self, text="Quit", command=self._root().destroy).grid(column=2, row=1)

        self.detector = DetectionAsDict(load='ckpt')

    def btn_DICOM_act(self):
        #dirname = tk.filedialog.askdirectory()
        #self.entry_DICOM_content.set(dirname)
        dcmname = tkinter.filedialog.askopenfilename()
        self.entry_DICOM_content.set(dcmname)

    def btn_output_act(self):
        dirname = tk.filedialog.askdirectory()
        self.entry_output_content.set(dirname)
    
    def btn_run_act(self):
        #dicom_folder = str(self.entry_DICOM_content.get())
        dicom_name = str(self.entry_DICOM_content.get())
        output_folder = str(self.entry_output_content.get())

        # Load DICOM
        self.label_status['text'] = "Loading"
        self.frm.update_idletasks()
        try:
            #topbed = TopBed(os.path.join(dicom_folder, 'dicom'))
            topbed = TopBed(dicom_name)
        except FileNotFoundError as e:
            tk.messagebox.showinfo(message='File Not Found', detail=e)
            self.label_status['text'] = "Fail"
            return
        except IOError as e:
            tk.messagebox.showinfo(message='IO Error', detail=e)
            self.label_status['text'] = "Fail"
            return

        # Evaluate
        self.label_status['text'] = "Running"
        self.frm.update_idletasks()
        bboxes = self.detector.eval(topbed.get_DX_intensity(), bbox=True)
        # interface
        if len(bboxes) == 0:
            save_path = os.path.join(output_folder, topbed.get_suggested_name())
            topbed.save_as_image(save_path, bboxes=bboxes)
            tk.messagebox.showinfo(
                message='Passed! Have a nice day.',
                detail='Screenshots saved to '+save_path+'.'
            )
        elif len(bboxes) >=10:
            new_bboxes_bottom=[]
            new_bboxes_top= []
            new_bboxes_middle=[]
            for bbox in bboxes:
                if 0.24*topbed.columns>bbox[1]:
                    new_bboxes_bottom.append(bbox)
                elif bbox[3] > 0.76 * topbed.columns:
                    new_bboxes_top.append(bbox)
                else:
                    new_bboxes_middle.append(bbox)
            if len(new_bboxes_bottom) >5 and len(new_bboxes_top)>5:
                if new_bboxes_middle:
                    save_path = os.path.join(output_folder, 'error_' + topbed.get_suggested_name())
                    topbed.save_as_image(save_path, bboxes=new_bboxes_middle)
                    save_path = os.path.join(output_folder, 'raw_' + topbed.get_suggested_name())
                    topbed.save_as_image(save_path, legend=False)
                    tk.messagebox.showerror(
                        message='Failed! You are on your own.',
                        detail='Results save in ' + save_path + '\nPlease double-check and save screenshots manually'
                    )
                else:
                    print('pass')
                    save_path = os.path.join(output_folder, topbed.get_suggested_name())
                    topbed.save_as_image(save_path, bboxes=new_bboxes_middle)
                    tk.messagebox.showinfo(
                        message='Passed! Have a nice day.',
                        detail='Screenshots saved to ' + save_path + '.'
                    )
            elif len(new_bboxes_bottom) >5 and len(new_bboxes_top)<5:
                if (new_bboxes_middle+new_bboxes_top):
                    save_path = os.path.join(output_folder, 'error_' + topbed.get_suggested_name())
                    topbed.save_as_image(save_path, bboxes=new_bboxes_middle+new_bboxes_top)
                    save_path = os.path.join(output_folder, 'raw_' + topbed.get_suggested_name())
                    topbed.save_as_image(save_path, legend=False)
                    tk.messagebox.showerror(
                        message='Failed! You are on your own.',
                        detail='Results save in ' + save_path + '\nPlease double-check and save screenshots manually'
                    )
                else:
                    save_path = os.path.join(output_folder, topbed.get_suggested_name())
                    topbed.save_as_image(save_path, bboxes=new_bboxes_middle)
                    tk.messagebox.showinfo(
                        message='Passed! Have a nice day.',
                        detail='Screenshots saved to ' + save_path + '.'
                    )
            elif len(new_bboxes_bottom)<5 and len(new_bboxes_top)>5:
                if (new_bboxes_middle + new_bboxes_bottom):
                    save_path = os.path.join(output_folder, 'error_' + topbed.get_suggested_name())
                    topbed.save_as_image(save_path, bboxes=new_bboxes_middle + new_bboxes_bottom)
                    save_path = os.path.join(output_folder, 'raw_' + topbed.get_suggested_name())
                    topbed.save_as_image(save_path, legend=False)
                    tk.messagebox.showerror(
                        message='Failed! You are on your own.',
                        detail='Results save in ' + save_path + '\nPlease double-check and save screenshots manually'
                    )
                else:
                    save_path = os.path.join(output_folder, topbed.get_suggested_name())
                    topbed.save_as_image(save_path, bboxes=bboxes)
                    tk.messagebox.showinfo(
                        message='Passed! Have a nice day.',
                        detail='Screenshots saved to ' + save_path + '.'
                    )
        else:
            print('failed')
            save_path = os.path.join(output_folder, 'error_' + topbed.get_suggested_name())
            topbed.save_as_image(save_path, bboxes=bboxes)
            save_path = os.path.join(output_folder, 'raw_' + topbed.get_suggested_name())
            topbed.save_as_image(save_path, legend=False)
            tk.messagebox.showerror(
                message='Failed! You are on your own.',
                detail='Results save in ' + save_path + '\nPlease double-check and save screenshots manually'
            )

        self.label_status['text'] = "Done"
        
if __name__ == "__main__":
    root = tk.Tk()
    root.title("Top Bed Defect Detection")
    app = App(root)
    root.columnconfigure(0, weight=1)
    root.rowconfigure(0, weight=1)
    root.mainloop()