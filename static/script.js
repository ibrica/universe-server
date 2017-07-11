     $(document).ready(() => {
         $('#trainBtn').click((e) => {
             //e.preventDefault();
             let model = $('#btnModel').val(),
                 env = $('#btnEnv').val();
             $.post('/train', { model, env })
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

        $(function(){
            $(".models li a").click(function(e){
              e.preventDefault();
              $("#btnModel").html($(this).text()+'<span class="caret"></span>');
              $("#btnModel").val($(this).attr("href"));
           });
        });

        $(function(){
            $(".envs li a").click(function(e){
              e.preventDefault();
              $("#btnEnv").html($(this).text()+'<span class="caret"></span>');
              $("#btnEnv").val($(this).attr("href"));
           });
        });
     });