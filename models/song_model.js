const mongoose = require("mongoose");
const Joi = require("joi");
const Schema = mongoose.Schema;

//define task model restrictions
const TimeStampSchema = new Schema (
{
    time: {
        type: String,
        required: true,
    },
    chordName: {
        type: String,
        required: true,
    },
    chordRoot: {
        type: String,
    },
    chordInterval: {
        type: String
    },
},
)
const SongSchema = new Schema(
  {
    name: {
      type: String,
      trim: true,
      required: true,
    },
    timestamp: {
      type: TimeStampSchema,
    },
    duration: {
      type: String,
      required: true,
    },
    key: {
      type: String,
      default: "C"
    },
    tag: {
        type: Number,
        required: true,
        default: 1
    },
    actist: {
      type: String,
    },
    url: {
      type: String,
      unique: true
    },
    image_url: {
      type: String,
    }
  },
  { collection: "Song" }
);

const UserSchema = new Schema (
  {
    userId: {
      type: String,
      unique: true,
    },
    userRate: {
      type: Number,
      required: true,
      default: 0,
    },
    email: {
      type: String,
      unique: true,
    }
  },
  {
    collection: "User",
  }
);
const EditionSchema = new Schema (
  {
    url: {
      type: String,
      required: true,
    },
    userId: {
      type: String,
      required: true,
    },
    key: {
      type: String,
      required: true,
    },
    timestamp: {
      type: TimeStampSchema,
    },
  },
  {
    collection: "Edition",
  }
);

const schema = Joi.object({
  name: Joi.string().trim(),
  description: Joi.string().trim(),
  createdDate: Joi.date(),
});

SongSchema.methods.joiValidation = function (songObject) {
  schema.required();
  return schema.validate(songObject);
};

const Song = mongoose.model("Song", SongSchema);
const User = mongoose.model("User", UserSchema);
const Edition = mongoose.model("Edition", EditionSchema);
module.exports = Song, User, Edition ;