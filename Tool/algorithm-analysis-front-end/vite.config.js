const path = require('path');

module.exports = {
    alias: {
        '@': path.resolve(__dirname, 'src'),
    },
    port: 8080,
    open: true,
    publicDir: 'public',
    proxy: {
        '/data': {
            target: 'http://10.10.1.203:8080/',
            changeOrigin: true,
            rewrite: (path) => path.replace(/^\/data/, '')
        }
    }
}
