$(document).ready(function(){
  $("#submit-form").submit(function(e){
    e.preventDefault();
    $('#submit-form').hide();
    // initiate the download
    $.ajax({
      url: "/dlprocess",
      type: "get",
      data: {
        url: $("#url-content").val(),
        format: 'bestvideo'
      },
      success: function(response) {
        $("#status-readout").html('Added to queue');
      },
      error: function(xhr) {
        //Do Something to handle error
      }
    });

    var interval = setInterval(function(){

        $.ajax({
          url: "/response",
          type: "get",
          data: {url: $("#url-content").val()},
          success: function(response) {
            if(response.includes('mp4')){
              $("#status-readout").html("Done. Click here to download.");
              $('#status-readout').wrap('<a href="./viddone/' + response + '" />');
              clearInterval(interval);
            }
            else {
              $("#status-readout").html(response);
            }

          },
          error: function(xhr) {
            //Do Something to handle error
          }
        }); },

      1000);  // check ever 1s
  });
});