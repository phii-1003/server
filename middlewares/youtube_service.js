const createError = require("http-errors");
const {google} = require('googleapis');
const path = require('path');

const youtube = google.youtube({
    version: 'v3',
});

const youtubeService = async (query) => {
    try {
        const response = await youtube.search.list({
            key: process.env.YOUTUBE_API_KEY,
            part: 'snippet',
            q: query,
            type: 'video',
            maxResults: 10,
        });
        
        const results = response.data.items;
        //convert reuslts to Song object
        const songs = results.map((result) => {
            return {
                _id : '',
                name: result.snippet.title,
                duration: '',
                tag: 1,
                actist: result.snippet.channelTitle,
                image_url: result.snippet.thumbnails.default.url,
                key: '',
                url: result.id.videoId,
            };
        });
        return songs;
    } catch (error) {           
        createError(error);
    }
}

module.exports = youtubeService;