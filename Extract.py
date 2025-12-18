import os
import shutil
import zipfile

def extract_data():
    if os.path.exists('F:\project\AI\Rock Paper Scissor.v2i.folder/'):
        shutil.rmtree('F:\project\AI\Rock Paper Scissor.v2i.folder/')
        print('F:\project\AI\Rock Paper Scissor.v2i.folder/ is removed!!!')
        


    with zipfile.ZipFile('F:\project\AI\Rock Paper Scissor.v2i.folder.zip','r') as target_file:
        target_file.extractall('F:\project\AI')
    
