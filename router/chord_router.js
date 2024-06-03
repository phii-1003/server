const chordRouter=require("express").Router();
const chordController = require("../controllers/chord_controller");
chordRouter.get("/reco",chordController.recognizeInput)
chordRouter.get("/reconex",chordController.recognizeAndExportInput)

module.exports=chordRouter