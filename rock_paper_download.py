from roboflow import Roboflow

rf = Roboflow(api_key="E8WIPjtVKD19G5AAvr5A")
project = rf.workspace("cpecgm3").project("rock-paper-scissor-p13xv")
version = project.version(2)
dataset = version.download("folder")