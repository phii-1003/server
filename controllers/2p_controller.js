const Song = require("../models/song_model");
const createError = require("http-errors");

const getSongCaculated = async (req, res, next) => {
    try {
        const song = await Song.findOne({ _id: req.query._id }, {}, { lean: true });
        return res.status(200).json(song);
    } catch (error) {
        next(createError(error));
    }
};

module.exports = {
    getSongCaculated
};