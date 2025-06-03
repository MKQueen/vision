const http = require('http');
const fs = require('fs');
const path = require('path');

const server = http.createServer((req, res) => {
    const filePath = req.url === '/' ? '/demo.html' : req.url;
    const fileExtension = path.extname(filePath);
    const contentType = {
        '.html': 'text/html',
        '.js': 'text/javascript',
        '.css': 'text/css',
        '.jpg': 'image/jpeg',
        '.png': 'image/png',
        '.ico': 'image/x-icon',
    }[fileExtension] || 'application/octet-stream';

    fs.readFile(__dirname + filePath, (err, data) => {
        if (err) {
            res.writeHead(404, { 'Content-Type': 'text/plain' });
            res.write('404 Not Found\n');
            res.end();
            return;
        }

        res.writeHead(200, { 'Content-Type': contentType });
        res.write(data);
        res.end();
    });
});

const PORT = process.env.PORT || 8000;

server.listen(PORT, () => {
    console.log(`Server running at http://198.18.0.1:${PORT}/`);
});
