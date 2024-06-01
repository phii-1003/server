// //we will be using express.js for creating our service
// const express = require("express");
// const cors = require("cors");
// const app = express();
// const mongoose = require("mongoose");
// var bodyParser = require("body-parser");
// const morgan = require("morgan");
// const helmet = require("helmet");
// const dotenv = require("dotenv");
// const songRoute = require("./router/song_router");
// const db = require("./database/mongoose_db");
// const errorMiddleWare = require("./middlewares/error_middleware");

// dotenv.config();

// app.use(bodyParser.json({limit:"50mb"}));
// app.use(helmet());
// app.use(cors());
// app.use(morgan("common"));
// db.connect();

// //ROUTES
// app.use("/api/v1/song", songRoute);
// // app.use("/v1/book", bookRoute);

// app.use(errorMiddleWare);

// //set available port to connect our server
// const PORT = process.env.PORT || 3000;
// app.listen(PORT, (err, suc) => {
//   if (err) throw err;
//   console.log(`Server running on ${PORT} port`);
// });