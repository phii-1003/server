const tomp3=require('yt-get');
const fs=require('fs');

const to_time_stamp=(json_str)=>{
    console.log(json_str)
    json_obj=JSON.parse(json_str)
    time_lst=[]
    chord_name_lst=[]
    chord_root_lst=[]
    interval_lst=[]
    json_obj.forEach((item,idx)=>{
        time_lst.push(item.Time)
        chord_name_lst.push(item.ChordName)
        chord_root_lst.push(item.ChordRoot)
        interval_lst.push(item.ChordInterval)
    })
    res_json={"Time":JSON.stringify(time_lst),"ChordName":JSON.stringify(chord_name_lst),"ChordRoot":JSON.stringify(chord_root_lst),"ChordInterval":JSON.stringify(interval_lst)}
    return JSON.stringify(res_json)
}

async function findAudio(videoURL){
    let videoTitle="";
    let bin_data;
    tomp3.getVideoMP3Base64(videoURL)
    .then((result) => {
        videoTitle=result.title;
        bin_data=result.base64;
        const buffer = Buffer.from(
            bin_data,  // only use encoded data after "base64,"
            'base64'
        )
        fs.writeFileSync('./input_data.mp3', buffer)
    })
    .catch((error) => {
        throw "Error in the link:" + error
    });
    
}   

const data_incoming_handler=async (python_process)=>{
    return new Promise((resolve,reject)=>{
        const handle_data=(data)=>{
            json_chord_data=Buffer.from(data).toString("utf8");
            // json_str=to_time_stamp(json_chord_data)
            json_str=json_chord_data
            python_process.stdout.removeListener('data',handle_data)
            // setTimeout(()=>{reject("Too long to recognize"),2000000})
            resolve(btoa(json_str))
            // resolve(json_str)
        }
        python_process.stdout.on('data',handle_data)
    })
}
const check_ready=async(python_process)=>{
    return new Promise((resolve,reject)=>{
        const ready_data=(data)=>{
            json_chord_data=Buffer.from(data).toString();
            if (json_chord_data=="Ready"){
                python_process.stdout.removeListener('data',ready_data)
                resolve(json_chord_data)
            }
            else{
                python_process.stdout.removeListener('data',ready_data)
                reject("Fail to run py model")
            }
        }
        python_process.stdout.on('data',ready_data)
    })
}
module.exports= {
    findAudio,
    data_incoming_handler,
    check_ready
}

