#epochs_related_functions

#from Dics Done 

import mne
import pandas as pd


#Epoch creation function - This function will create epochs from the raw data based on the events found in the "STI101" channel, with a time window from -0.9s to 1.5s around each event.
def create_raw_epochs(raw): 

    events = mne.find_events(
        raw,
        stim_channel="STI101",
        shortest_event=1,
        min_duration=0.001,
    )

    stim_events = events[(events[:, 2] >= 1) & (events[:, 2] <= 80)]

    epochs = mne.Epochs(
        raw,
        stim_events,
        tmin=-0.9,
        tmax=1.5,    
        preload=False
    )   
    return epochs, events

#Metadata creation function for epochs - This function will create a metadata dataframe based on the events and their codes, and then attach it to the epochs object. 
def create_metadata(epochs, events):

    #Category (faces, objects, fonts, false_fonts)
    def category(event_id):
        if 1 <= event_id <= 20:
            return "faces"
        elif 21 <= event_id <= 40:
            return "objects"
        elif 41 <= event_id <= 60:
            return "fonts"
        elif 61 <= event_id <= 80:
            return "false_fonts"
        else:
            return None
        
    #Orientation (center, left, right)
    def orientation(event_id):
        mapping = {
            101: "center",
            102: "left",
            103: "right",
        }
        return mapping.get(event_id, None)
    
    #Duration (500ms, 1000ms, 1500ms)
    def duration(event_id):
        mapping = {
            151: "dur_500ms",
            152: "dur_1000ms",
            153: "dur_1500ms",
        }
        return mapping.get(event_id, None)
    
    #Relevance (target, relevant, irrelevant)
    def relevance(event_id):
        mapping = {
            201: "target",
            202: "relevant",
            203: "irrelevant",
        }
        return mapping.get(event_id, None)
    
    #Sex of the face (faces_man, faces_woman)
    def sex(event_id):
        if 1 <= event_id <= 10:
            return "faces_man"
        elif 11 <= event_id <= 20:
            return "faces_woman"
        else:
            return None
        
    #Creating metadata for epochs
    metadata_rows = []

    for stim in epochs.events:
        stim_sample = stim[0]
        stim_code = stim[2]
        
        window = events[(events[:,0] > stim_sample) & 
                        (events[:,0] < stim_sample + 200)]

        ori = None
        dur = None
        rel = None

        for e in window:
            if e[2] in [101,102,103]:
                ori = orientation(e[2])
            elif e[2] in [151,152,153]:
                dur = duration(e[2])
            elif e[2] in [201,202,203]:
                rel = relevance(e[2])

        metadata_rows.append({
            "sti_id": stim_code,
            "category": category(stim_code),
            "orientation": ori,
            "duration": dur,
            "relevance": rel,
            "sex": sex(stim_code) if category(stim_code) == "faces" else None
        })

    metadata = pd.DataFrame(metadata_rows)
    epochs.metadata = metadata

    return epochs, metadata



