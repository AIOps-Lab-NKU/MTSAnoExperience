import {ElLoading} from 'element-plus'; //项目已经全局引入element的话可以不单独引入


let loading        //定义loading变量

function startLoading() {    //使用Element loading-start 方法
    loading = ElLoading.service({
        lock: true,
        text: '加载中……',
        fullscreen: true,
        background: 'rgba(0, 0, 0, 0.7)'
    })
}

function endLoading() {    //使用Element loading-close 方法
    loading.close()
}

//那么 showFullScreenLoading() tryHideFullScreenLoading() 要干的事儿就是将同一时刻的请求合并。
//声明一个变量 needLoadingRequestCount，每次调用showFullScreenLoading方法 needLoadingRequestCount + 1。
//调用tryHideFullScreenLoading()方法，needLoadingRequestCount - 1。needLoadingRequestCount为 0 时，结束 loading。
let needLoadingRequestCount = 0

export function showFullScreenLoading() {
    if (needLoadingRequestCount === 0) {
        startLoading()
    }
    needLoadingRequestCount++
}

export function tryHideFullScreenLoading() {
    if (needLoadingRequestCount <= 0)
        return
    needLoadingRequestCount--
    if (needLoadingRequestCount === 0) {
        endLoading()
    }
}
