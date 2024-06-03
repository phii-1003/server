## Application Server for DATN

Official run (it is required ro run in Linux enviroment):
1a. create data by run file CNN/data_gen.py
2a. for training: change option of main func in CNN/main_CNN.py to training; change to validating if needs data validation
1b. for whole recognizing module: get the file and the groundtruth of main folder( if you need accuracy), then run main.py

Note: 
1. data type for chord server
const data_req_type={
    body:{
        isDataURL: Boolean,
        data:Uint8Array,
    }
}
const data_res_type_recog={
    body:{
        status: Number,
        isError:Boolean,
        error:String, //empty if no error
        chord_result: TimeStamp,
    }
}
const data_res_type_reco_and_export={
    body:{
        status: Number,
        isError:Boolean,
        error:String, //empty if no error
        chord_result: TimeStamp,
        export_result: Uint8Array
    }
}
2. Export audio and json_chord is formatted as base64
3. Time in chord json is in ms