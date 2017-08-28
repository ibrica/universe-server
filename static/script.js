     $(document).ready(() => {
         $('#trainBtn').click((e) => {
             //e.preventDefault();
             let model = $('#btnModel').val(),
                 env = $('#btnEnv').val(),
                 workers = $('#btnWorkers').val();
             $.post('/train', { model, env, workers })
                 .done((data) => {
                     $("#alerts").append('<div class="alert alert-success"> <strong>Success!</strong> You started training your environment! </div>');
                 })
                 .fail(function() {
                     $("#alerts").append('<div class="alert alert-danger"><strong>Error!</strong> Training couldn\'t be started! </div>');
                 });
         });
         $('#playBtn').click(() => {
             let model = $('#btnModel').val(),
                 env = $('#btnEnv').val();
             $.post('/play', { model, env })
                 .done((data) => {
                     $("#alerts").append('<div class="alert alert-success"> <strong>Success!</strong> You started playing your environment! </div>');
                 })
                 .fail(function() {
                     $("#alerts").append('<div class="alert alert-danger"><strong>Error!</strong> Playing couldn\'t be started! </div>');
                 });
         });

         $(function() {
             $(".models li a").click(function(e) {
                 e.preventDefault();
                 $("#btnModel").html($(this).text() + '<span class="caret"></span>');
                 $("#btnModel").val($(this).attr("href"));
             });
         });

         $(function() {
             $(".envs li a").click(function(e) {
                 e.preventDefault();
                 $("#btnEnv").html($(this).text() + '<span class="caret"></span>');
                 $("#btnEnv").val($(this).attr("href"));
             });
         });
         $(function() {
             $(".workers li a").click(function(e) {
                 e.preventDefault();
                 $("#btnWorkers").html($(this).text() + '<span class="caret"></span>');
                 $("#btnWorkers").val($(this).attr("href"));
             });
         });
         window.chartColors = {
             red: 'rgb(255, 99, 132)',
             orange: 'rgb(255, 159, 64)',
             yellow: 'rgb(255, 205, 86)',
             green: 'rgb(75, 192, 192)',
             blue: 'rgb(54, 162, 235)',
             purple: 'rgb(153, 102, 255)',
             grey: 'rgb(201, 203, 207)'
         };
         let ctx = $("#rewardChart");
         $(function() {
             setInterval(() => {
                 let model = $('#btnModel').val();
                 $.get(`/data/${model}`, function(data) {
                     if (!data) return;
                     let averageReward = data.averageReward,
                         minReward = data.minReward,
                         maxReward = data.maxReward,
                         episodeCounter = data.episodeCounter;

                     let config = {
                         type: 'line',
                         data: {
                             labels: episodeCounter,
                             datasets: [{
                                 label: "Minimal reward",
                                 backgroundColor: window.chartColors.red,
                                 borderColor: window.chartColors.red,
                                 data: minReward,
                                 fill: false,
                             }, {
                                 label: "Average reward",
                                 fill: false,
                                 backgroundColor: window.chartColors.blue,
                                 borderColor: window.chartColors.blue,
                                 data: averageReward
                             }, {
                                 label: "Maximal reward",
                                 fill: false,
                                 backgroundColor: window.chartColors.purple,
                                 borderColor: window.chartColors.purple,
                                 data: maxReward
                             }]
                         },
                         options: {
                             responsive: true,
                             title: {
                                 display: true,
                                 text: 'Rewards chart'
                             },
                             tooltips: {
                                 mode: 'index',
                                 intersect: false,
                             },
                             hover: {
                                 mode: 'nearest',
                                 intersect: true
                             },
                             scales: {
                                 xAxes: [{
                                     display: true,
                                     scaleLabel: {
                                         display: true,
                                         labelString: 'Episodes'
                                     }
                                 }],
                                 yAxes: [{
                                     display: true,
                                     scaleLabel: {
                                         display: true,
                                         labelString: 'Rewards'
                                     },
                                     ticks: {
                                         beginAtZero: true
                                     }
                                 }]
                             }
                         }
                     };

                     let chart = new Chart(ctx, {
                         type: config.type,
                         data: config.data,
                         options: config.options
                     });

                 });
             }, 60000); //Every minute new update of the chart
         })
     });