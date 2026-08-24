from pathlib import Path
import shutil
import zipfile

def extract_data():
    project_dir = Path('F:/project/AI')
    extract_folder = project_dir / 'Rock Paper Scissor.v2i.folder'
    zip_file = project_dir / 'Rock Paper Scissor.v2i.folder.zip'

    if extract_folder.exists():
        shutil.rmtree(extract_folder)
        print(f'{extract_folder} is removed!!!')

    with zipfile.ZipFile(zip_file, 'r') as target_file:
        target_file.extractall(project_dir)
    
