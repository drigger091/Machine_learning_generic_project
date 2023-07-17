from setuptools import find_packages ,setup

from typing import List

hypen_E_DOT = '-e .'

def get_requirements(file_path:str)->List[str]:
        ''' This function will return the list of requirememtx
        '''
        requirements = []
        with open(file_path) as file_obj:
            requirements = file_obj.readlines()
            requirements =[req.replace("\n","") for req in requirements]

            if hypen_E_DOT in requirements:
                  requirements.remove(hypen_E_DOT)

        
        return requirements
                  
setup(
    name='ML_project',
    version='0.0.1',
    author="Dipayan",
    author_email='cntctdipayan@gmail.com',
    packages= find_packages(),
    install_requires = get_requirements('requirements.txt')
)

