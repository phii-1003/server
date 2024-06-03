const createError = require("http-errors");
const utils = require('../utils/utils')
const fs = require('fs')
const savefile = require('save-file')
const spawner = require("child_process").spawn

const python_process = spawner("python", ["./main.py"])
python_process.on('close', () => { console.log('closed') })
let py_ready_flag = false
const recognizeInput = async (req, res, next) => {
    let data = req.query;
    let isURL = data.isDataURL;
    let audio_buffer = data.data;
    console.log(data)
    try {
        //create audio data file
        if (isURL)
            await utils.findAudio(audio_buffer);
        else
            await savefile.save(audio_buffer, "input_data.mp3");
        console.log("Audio file created!");
        req_json = JSON.stringify({ "willExport": false })
        if (!py_ready_flag) {
                await utils.check_ready(python_process);
                console.log("Py model Ready")
                py_ready_flag = true
            }
        python_process.stdin.cork()
        python_process.stdin.write(req_json + "\n")
        python_process.stdin.uncork()

        //handle data from stdout
        let json_chord_data = await utils.data_incoming_handler(python_process);
        console.log(json_chord_data)
        return res.status(201).json({
            isError: false,
            error: "",
            chord_result: json_chord_data,
        });

    } catch (err) {
        next(createError(err))
    }

};

const recognizeAndExportInput = async (req, res, next) => {
    let data = req.query;
    let isURL = data.isDataURL;
    let audio_buffer = data.data;
    try {
        //create audio data file
        if (isURL)
            await utils.findAudio(audio_buffer);
        else
            await savefile.save(audio_buffer, "input_data.mp3");
        console.log("Audio file created!");
        req_json = JSON.stringify({ "willExport": true })
        if (!py_ready_flag) {
            await utils.check_ready(python_process);
            console.log("Py model Ready")
            py_ready_flag = true
        }
        python_process.stdin.cork()
        python_process.stdin.write(req_json + '\n')
        python_process.stdin.uncork()

        //handle data from stdout
        let json_chord_data = await utils.data_incoming_handler(python_process);
        //load exported audio
        let export_res = fs.readFileSync('./tmp.mp3', 'base64')
        return res.status(201).json({
            isError: false,
            error: "",
            chord_result: json_chord_data,
            export_result: export_res
        });

    } catch (err) {
        next(createError(err))
    }

};

module.exports = {
    recognizeInput,
    recognizeAndExportInput
};



