const Song = require("../models/song_model");
const createError = require("http-errors");

//add new task
const addNewSong = async (req, res, next) => {
  const newSong = new Song(req.body);

  const { error } = newSong.joiValidation(req.body);

  if (error) {
    return res.status(400).json({
      result: false,
      message: error,
    });
  } else {
    try {
      const result = await newSong.save();
      if (result) {
        return res.status(201).json({
          result: true,
          message: "New task saved.",
        });
      } else {
        return res.status(400).json({
          result: false,
          message: "Something went wrong while saving task.",
        });
      }
    } catch (e) {
      next(createError(e));
    }
  }
};

//update task
const updateSong = async (req, res, next) => {
  try {
    const task = await Song.findById(req.body.id, {}, { lean: true });
    if (task) {
      const willBeUpdated = await Song.findByIdAndUpdate(
        { _id: req.body.id },
        req.body,
        { lean: true, new: true }
      );
      if (willBeUpdated) {
        return res.status(201).json({
          result: true,
          message: "Song updated.",
        });
      } else {
        return res.status(400).json({
          result: true,
          message: "Something went wrong while updating task.",
        });
      }
    } else {
      return res.status(404).json({
        result: false,
        message: "No record found.",
      });
    }
  } catch (error) {
    next(createError(error));
  }
};

//delete task
const deleteSong = async (req, res, next) => {
  try {
    const task = await Song.findByIdAndDelete({ _id: req.body.id });
    if (task) {
      return res.status(201).json({
        result: true,
        message: "Song deleted.",
      });
    } else {
      return res.status(400).json({
        result: false,
        message: "Something went wrong while deleting task.",
      });
    }
  } catch (error) {
    next(createError(error));
  }
};

//get all data
const getAllSongs = async (req, res, next) => {
  try {
    const allData = await Song.find({}, {}, { lean: true });
    return res.status(200).json(allData);
  } catch (error) {
    next(createError(error));
  }
};

// <<<<<<<<<<<<<<  ✨ Codeium Command 🌟 >>>>>>>>>>>>>>>>
/**
// Retrieves all songs with the given name.
/*/
const getSongByName = async (req, res, next) => {
  console.log(req.query.name);
  try {
  // Find all songs with the given name.
    const allData = await Song.find({ name: { $regex: `${req.query.name}`, $options: 'i' }});
  // Return the songs as a JSON response.
    return res.status(200).json(allData);
  } catch (error) {
  // If an error occurs, pass it to the error middleware.
    next(createError(error));
  }
};

const getSong = async (req, res, next) => {
  try {
    const song = await Song.findOne({ _id: req.query._id}, {}, { lean: true });
    return res.status(200).json(song);
  } catch (error) {
    next(createError(error)); 
  }
};

module.exports = {
  addNewSong,
  updateSong,
  deleteSong,
  getAllSongs,
  getSongByName,
  getSong
};