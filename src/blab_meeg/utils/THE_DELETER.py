# %%

def the_deleter(out_paths, folder):    

    for old_file in out_paths[folder].glob("*.fif"):
        old_file.unlink()    
        print(f"Deleted: {old_file.name}")

    



