import os, sys
import time

import pydicom
import numpy as np
import PIL
import PIL.Image
import PIL.ImageDraw


class TopBed(object):
    def __init__(self, dcm_path):
        super().__init__()
        self.load(dcm_path)
        self.rows = self.dcm.Rows
        self.columns = self.dcm.Columns

    def load(self, dcm_path):
        self.dcm = pydicom.dcmread(dcm_path)
        #self.rows = self.dcm.Rows
        #self.columns = self.dcm.Columns
        #print(self.dcm.Rows,self.dcm.Columns)

    def get_DX_intensity(self):
        # get X-Ray beam intensity incident on the detector
        #assert self.dcm.PixelIntensityRelationship == 'LIN'
        array = self.dcm.pixel_array * float(self.dcm.RescaleSlope) + float(self.dcm.RescaleIntercept)
        return array.astype(np.float64)
    
    def get_suggested_name(self):
        return str(self.dcm.PatientName)+'_'+str(self.dcm.SeriesNumber)+'.png'

    def save_as_image(self, img_path, bboxes=None, legend=True):
        WL_WW_map = {1:(-49, 40), 2:(-45, 40), 3:(-45,40),4:(-45,40)}
        font_size = 20
        font_padding = 5
        fill_color = (65,255,0)
        bbox_color = (255, 0, 0)
        # assert int(self.dcm.InstanceNumber) in WL_WW_map.keys(), 'Unknown InstanceNumber'
        #print(self.dcm)
        #print([int(self.dcm.SeriesNumber)])
        WL, WW = WL_WW_map[int(self.dcm.SeriesNumber)]
        array = self.get_DX_intensity()
        array = (array - (WL - WW/2)) / WW
        array = np.clip(array*255, 0 , 255).astype(np.uint8)
        img = PIL.Image.fromarray(array).convert('RGB')
        if legend:
            draw = PIL.ImageDraw.Draw(img)
        else:
            draw = PIL.ImageDraw.Draw(img.copy())
        font = PIL.ImageFont.truetype("./FreeMono.ttf", font_size)

        if bboxes is not None:
                for bbox in bboxes:
                    draw.rectangle(bbox, fill=None, outline=bbox_color, width=1)

        draw.text(
            (font_padding, font_padding),
            text=str(self.dcm.PatientName)+' ('+str(self.dcm.SeriesNumber)+')',
            font=font, fill=fill_color)
            #font=font, stroke_fill=255, fill=0, stroke_width=0)

        draw.text(
            (font_padding, font_padding+font_size),
            text='W/C: '+str(WW)+'/'+str(WL),
            font=font, fill=fill_color)
            #font=font, fill=(255,69,0))

        content_datetime = str(self.dcm.ContentDate)+str(self.dcm.ContentTime)
        # with fractional part (< 1 second) ignored
        content_datetime = content_datetime.split('.')[0]
        content_datetime = content_datetime.ljust(8+6, '0')
        #if len(content_datetime) == 1:
        #    content_datetime.append['0']
        content_datetime = time.strptime(content_datetime, '%Y%m%d%H%M%S')
        #content_date = ':'.join(map(''.join, zip(*[iter(content_date)]*2)))
        draw.text(
            (font_padding, font_padding+font_size*2),
            text=time.strftime('%Y-%m-%d', content_datetime),
            font=font, fill=fill_color)

        #content_time = str(self.dcm.ContentTime).split('.')[0]
        #assert len(content_time) <= 6
        #content_time = content_time.ljust(6, '0')
        #content_time = ':'.join(map(''.join, zip(*[iter(content_time)]*2)))
        draw.text(
            (font_padding, font_padding+font_size*3),
            text=time.strftime('%H:%M:%S', content_datetime),
            font=font, fill=fill_color)

        img.save(img_path)

if __name__ == "__main__":
    d = TopBed(sys.argv[1])
    d.save_as_image('img.png')