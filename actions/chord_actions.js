const createError = require("http-errors");
import { create } from "../datatype/song_datatype";
import { findAudio } from "../utils/utils";
const spawner=require("child_process").spawn
const python_process=spawner("python",["./main.py"])
python_process.stdout.on('data',(data)=>{
    
})
const recognizeInput=async (req,res,next)=>{
    let data=req.body;
    isURL,audio_buffer,willExport=data.isDataURL,data.data,data.willExport
    try{
        //create audio data file
        if (isURL)
            await findAudio(audio_buffer);
        else
            await save(audio_buffer,"input_data.mp3");
        console.log("Audio file created!");
        req_json=JSON.stringify({"willExport": willExport})
        python_process.stdin.cork()
        python_process.stdin.write(req_json+'\n')
        python_process.stdin.uncork()
        return res.status(201).json({
            result: true,
            message: "Song updated.",
          });

    } catch(err){
        next(createError(err))
    }
    
}

// const createInputAsFile=async (req,res,next)=>{
//     let data=req.body;
    
//     try{
//         await save(data,"input_data.mp3");
//         console.log("Audio file created!");
//     } catch(err){
//         next(createError(err))
//     }
    
// }
// const createInputAsURL=async(req,res,next)=>{
//     let url=req.body;
//     try{
//         await findAudio(url);
//         console.log("Audio file created!");
//     } catch(err){
//         next(createError(err));
//     }
// }



