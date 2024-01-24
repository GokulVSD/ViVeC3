import os
import shutil


# Directory Management Utilities
# Directory Creation
def create_directory(directory_path, override=True):
    if os.path.exists(directory_path):
        print("Directory {} already Exists!".format(directory_path))
        if override:
            delete_directory(directory_path)
            return create_directory(directory_path)
        created_directory = False
    else:
        try:
            os.mkdir(directory_path)
            created_directory = True
        except Exception as e:
            print("Exception in Creating Directory:", e)
            created_directory = False
    return created_directory


# Directory Management Utilities
# Directory Deletion
def delete_directory(directory_path):
    try:
        if os.path.exists(directory_path):
            shutil.rmtree(directory_path)
            deleted_directory = True
        else:
            print("Directory Path {} does not exist!".format(directory_path))
            deleted_directory = True
    except Exception as e:
        print("Exception in Deleting Directory:", directory_path, e)
        deleted_directory = True
    return deleted_directory