import os, sys
import os.path
import numpy as np
import tqdm
import imageio
import pydicom
import pydicom.fileset

def test_dicom(path):
    try:
        pydicom.dcmread(path)
    except pydicom.errors.InvalidDicomError:
        return False
    return True

def main():
    print('Indexing DICOM files...')
    dcmfiles = []
    for root, dirs, files in os.walk(sys.argv[1]):
        files = [os.path.join(root, file) for file in files]
        files = filter(test_dicom, files)
        dcmfiles.extend(files)
    
    fs = pydicom.fileset.FileSet()
    for file in dcmfiles:
        fs.add(file)
    
    '''
    ind = {}
    for instance in fs:
        instance = instance.load()
        if instance.StudyInstanceUID not in ind:
            ind[instance.StudyInstanceUID] = {}
        assert instance.SeriesInstanceUID not in ind[instance.StudyInstanceUID]
        ind[instance.StudyInstanceUID][instance.SeriesInstanceUID] = instance.SeriesDescription
    '''
    
    if len(sys.argv) <= 2:
        return
    
    print('Exporting to '+sys.argv[2]+'...')
    for instance in tqdm.tqdm(fs):
        instance = instance.load()
        assert instance.SeriesDescription in ['Topogram  0.6  T20f', 'Dose Report', 'Patient Protocol']
        if instance.SeriesDescription != 'Topogram  0.6  T20f':
            continue
        save_folder = os.path.join(sys.argv[2], instance.SeriesInstanceUID)
        os.mkdir(save_folder)
        WL = -49
        WW = 40
        array = instance.pixel_array * float(instance.RescaleSlope) + float(instance.RescaleIntercept)
        array = (array - (WL-WW/2))/WW
        pydicom.dcmwrite(os.path.join(save_folder, 'dicom'), instance)
        imageio.imsave(os.path.join(save_folder, 'image.png'), \
                np.clip(array*255, 0, 255).astype(np.uint8))

if __name__ == '__main__':
    main()