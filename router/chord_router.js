const chordRouter=require("express").Router();
const chordController = require("../actions/chord_actions");
chordRouter.get("/reco",chordController.recognizeInput)
chordRouter.get("/reconex",chordController.recognizeAndExportInput)

module.exports=chordRouter