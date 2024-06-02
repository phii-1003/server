const songRouter = require("express").Router();
const songController = require("../actions/song_actions");

// songRouter.post("/addNewSong", songController.addNewSong);

// songRouter.post("/updateSong", songController.updateSong);

// songRouter.post("/deleteSong", songController.deleteSong);

songRouter.get("/getAllSongs", songController.getAllSongs);
songRouter.get("/getSongByName", songController.getSongByName);
songRouter.get("/getSong", songController.getSong);

module.exports = songRouter;