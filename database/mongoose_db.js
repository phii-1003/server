//local database connection
const mongoose = require("mongoose");
const dbConnection = "mongodb+srv://dbUser:dbUserPassword@atlascluster.llmc0m2.mongodb.net/";

// mongoose
//   .connect(dbConnection)
//   .then((suc) => console.log("Connected to db"))
//   .catch((err) => console.log("Error occurred while connecting to db", err));

async function connect() {
  try {
    await mongoose.connect(dbConnection)
    .then((suc) => console.log("Connected to db"))
  } catch (error) {
    console.log("Error occurred while connecting to db", err);
  }
}
module.exports = { connect }