import {createRouter, createWebHashHistory} from "vue-router";

const routes = [
    {path: "/", redirect: "/home"},
    {
        path: "/home",
        name: "home",
        component: () => import("../pages/homePage.vue")
    },
    {
        path: "/test",
        name: "test",
        component: () => import("../pages/generateView.vue")
    }
]

const router = createRouter({
    history: createWebHashHistory(),
    routes: routes
})

export {router}
