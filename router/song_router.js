const songRouter = require("express").Router();
const songController = require("../controllers/song_controller");

// songRouter.post("/addNewSong", songController.addNewSong);

// songRouter.post("/updateSong", songController.updateSong);

// songRouter.post("/deleteSong", songController.deleteSong);

songRouter.get("/getAllSongs", songController.getAllSongs);
songRouter.get("/getSongByName", songController.getSongByName);
songRouter.get("/getSong", songController.getSong);

module.exports = songRouter;