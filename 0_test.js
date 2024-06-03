// json_obj=[{"helllo":5,"rt":'tt'},{"45":"hadd","Ed":-1}]
// str_json=JSON.stringify(json_obj)
// res_obj={"isDataURL":true,"data":"https://www.youtube.com/watch?v=qAlyjGrThGo"}

// console.log(JSON.stringify(res_obj))
const tomp3=require('yt-get');
const fs=require('fs')

const func=async()=>{
    tomp3.getVideoMP3Base64("https://www.youtube.com/watch?v=qAlyjGrThGo")
    .then((result) => {
        console.log(result)
        videoTitle=result.title;
        wavUrl=result.base64;
        const buffer = Buffer.from(
            wavUrl,  // only use encoded data after "base64,"
            'base64'
          )
        fs.writeFileSync('./input_data.mp3', buffer)
        // console.log(buffer)
    })
    .catch((error) => {
        throw "Error in the link: " + error
    });
}
func()





