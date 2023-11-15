<template>
  <div class="main-content">
    <el-table
        :data="tableData"
        border
        :summary-method="getAlgo"
        show-summary
        style="width: 100%">
      <el-table-column prop="algorithm" label="algorithm"/>
      <el-table-column prop="precision" label="precision" sortable/>
      <el-table-column prop="recall" label="recall" sortable/>
      <el-table-column prop="f1 score" label="f1 score" sortable/>
    </el-table>
    <el-input v-model="algoSort" readonly>
      <template #prepend>算法排序</template>
    </el-input>
    <template v-for="(item,index) in getMiddleData" :key="index">
      <div class="each-item">
        <h4>{{ item.title }}</h4>
        <span>p:{{ formatScore(item.p) }}&nbsp;&nbsp;r:{{ formatScore(item.r) }}&nbsp;&nbsp;f1:{{
            formatScore(item.f1)
          }}</span>
        <div class="show">
          <select-item :index="index"/>
          <div class="echarts_show">
          </div>
        </div>
      </div>
    </template>
  </div>
</template>
<script>
import selectItem from "./selectItem.vue"
import {getCSVDataForm} from "../js/utils"


export default {
  name: "selectMiddle",
  components: {
    selectItem
  },
  data() {
    return {
      tableData: [],
      algoSort: '',
      graphics: []
    }
  },
  methods: {
    formatScore(score) {
      if (!score && score !== 0) {
        return 'none'
      }
      return (score * 100).toFixed(4) + '%'
    },
    showEcharts() {
      for (let i = 0; i < this.graphics.length; i++) {
        this.graphics[i].destroy()
      }
      this.graphics.length = 0
      const showDivs = document.getElementsByClassName("echarts_show");
      for (let i = 0; i < showDivs.length; i++) {
        const [data, max, min] = getCSVDataForm(this.getMiddleData[i]['score'], this.$store.state.timesTrack,
            null, this.getMiddleData[i]['title'], this.getMiddleData[i]['threshold'])
        this.graphics.push(new this.$dygraphs(showDivs[i], data, {
          legend: "always",
          axis: {
            y: {
              valueParser: function (x) {
                return x.toPrecision(2)
              },
            }
          },
          drawGrid: false,
          valueRange: [min, max],
          includeZero: false,
          colors: ['green', 'red', 'blue'],
          // showRangeSelector: true,
          legendFormatter: function (data) {
            // console.log(data)
            if (data.x) {
              return `<span class="legend-div" style="position:absolute;
                        background-color: aquamarine;
                        width: fit-content;
                        height: fit-content;
                        z-index: 999;
                        top:${window.event.offsetY + 14}px;
                        left:${window.event.offsetX + 10}px">${data.x}</br> 分数:${data.series[0].yHTML ? data.series[0].yHTML : data.series[1].yHTML}
                        </br> 阈值:${data.series[2].yHTML}</span>`
            } else {
              return ''
            }
          }
        }))
      }
    },
    loadTableData() {
      this.tableData.length = 0
      for (let i = 0; i < this.getMiddleData.length; i++) {
        this.tableData.push({
          'algorithm': this.getMiddleData[i]['title'],
          'recall': this.getMiddleData[i]['r'],
          'precision': this.getMiddleData[i]['p'],
          'f1 score': this.getMiddleData[i]['f1'],
        })
      }
    },
    getAlgo(param) {
      const {data} = param
      this.algoSort = ''
      for (let i = 0; i < data.length; i++) {
        this.algoSort = this.algoSort + data[i].algorithm + " "
      }
      return []
    },
  },
  computed: {
    getMiddleData() {
      return this.$store.state.middleData
    }
  },
  watch: {
    getMiddleData: {
      deep: true,
      handler(val) {
        this.loadTableData()
        this.$nextTick(() => {
          this.showEcharts();
        })
      }
    }
  }
}
</script>

<style scoped>
.echarts_show {
  width: 94%;
  height: 200px;
}

.main-content {
  display: flex;
  flex-direction: column;
}

.each-item {
  display: flex;
  align-items: center;
  justify-content: center;
  flex-direction: column;
}

.show {
  display: flex;
  width: 100%;
  align-items: center;
  justify-content: center;
}

h4 {
  margin-block-start: 0.5em;
  margin-block-end: 0.5em;
}
</style>
