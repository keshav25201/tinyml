import os
def get_dir_size(path='.'):
    total = 0
    with os.scandir(path) as it:
        for entry in it:
            if entry.is_file():
                total += entry.stat().st_size
            elif entry.is_dir():
                total += get_dir_size(entry.path)
    return total
normal_model_size = get_dir_size('my_model')
print("normal model is %d bytes" % normal_model_size)
tinyml_model_size = os.path.getsize("fmnist.tflite")
print("tinyml model is %d bytes" % tinyml_model_size)
difference = normal_model_size - tinyml_model_size
print("Difference is %d bytes" % difference)