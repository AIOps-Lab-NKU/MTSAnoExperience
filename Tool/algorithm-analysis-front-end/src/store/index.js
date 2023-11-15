import {createStore} from 'vuex'

let store = createStore({
        state() {
            return {
                selected: new Set(),
                middleData: [],
                rightData: [],
                lastChangeSelect: -1,
                labels: [],
                timesTrack: [],
                datasetName: 'water-distribution',
                isCleared: true,
                isShowAll: false
            }
        },
        mutations: {
            clearConstant(state) {
                state.selected = new Set()
                state.labels.length = 0
                state.middleData.length = 0
                state.rightData.length = 0
                state.lastChangeSelect = 0
                state.timesTrack.length = 0
                state.isCleared = true
                state.datasetName = 'water-distribution'
            },
            addSelect(state, index) {
                state.selected.add(index);
                state.lastChangeSelect = index
                state.isCleared = false
            },
            delSelect(state, index) {
                state.selected.delete(index);
                state.lastChangeSelect = index
                state.isCleared = false
            },
            setDatasetName(state, data) {
                state.datasetName = data
            },
            setLabels(state, data) {
                state.labels = data
            },
            setTimeTrack(state, value) {
                state.timesTrack = value
            },
            setMiddleData(state, data) {
                state.middleData = data
            },
            setRightData(state, data) {
                state.rightData = data
            },
            setShowAll(state, data) {
                state.isShowAll = data
            }
        }
    }
);
export default store;
