import { getVideoMP3Binary } from "yt-get";
import save from "save-file";
async function findAudio(videoURL){
    let videoTitle="";
    let bin_data;
    getVideoMP3Binary(videoURL)
    .then((result) => {
        const { mp3, title } = result;
        videoTitle=title;
        bin_data=mp3;
    })
    .catch((error) => {
        throw "Error in the link:" + error
    });
    await save(bin_data,"input_data.mp3")
}   
export {
    findAudio,
}