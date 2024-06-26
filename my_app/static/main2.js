var audio;
var vol_btn = document.getElementById("volume-button");
vol_btn.setAttribute("src", "/static/vol_on.png");

var retry_button = document.getElementById("retry-button");
retry_button.setAttribute("src", "/static/retry.png");

var current_vol = 0;
var is_on = 1;

//Hide Pause Initially
$("#pause").hide();

var ul = document.querySelector("ul");
for (var i = ul.children.length; i >= 0; i--) {
  ul.appendChild(ul.children[(Math.random() * i) | 0]);
}

// Ensure the document is fully loaded before initializing the audio
$(document).ready(function () {
  initAudio($("#playlist li:first-child"));
});

function initAudio(element) {
  var song = element.attr("song_url");
  var title = element.text();
  var cover = element.attr("cover");
  var movie = element.attr("movie");

  //Create a New Audio Object
  audio = new Audio(song);

  if (!audio.currentTime) {
    $("#duration").html("0:00");
  }

  // set title and movie
  $("#audio-player #title").text(title);
  $("#audio-player #movie").text(movie);

  //Insert Cover Image
  $("#audio-image #cover").attr("src", "/static/" + cover);

  $("#playlist li").removeClass("active");
  element.addClass("active");

  //After song ends play next song
  $(audio).on("ended", function () {
    $(".progress-bar").css("width", 0 + "%");
    $("#next").trigger("click");
  });

  // Play the audio automatically
  audio.play();
  $("#play").hide();
  $("#pause").show();
  showDuration();
}

//Play Button
$("#play").click(function () {
  audio.play();
  $("#play").hide();
  $("#pause").show();
  $("#duration").fadeIn(400);
  $(".progress").fadeIn(400);
  showDuration();
});

//Pause Button
$("#pause").click(function () {
  audio.pause();
  $("#pause").hide();
  $("#play").show();
});

//Stop Button
$("#stop").click(function () {
  audio.pause();
  audio.currentTime = 0;
  $("#pause").hide();
  $("#play").show();
  $("#duration").fadeOut(400);
  $(".progress").fadeOut(400);
});

//Next Button
$("#next").click(function () {
  audio.pause();
  var next = $("#playlist li.active").next();
  if (next.length == 0) {
    next = $("#playlist li:first-child");
  }
  if ($("#play").is(":visible")) {
    $("#play").hide();
    $("#pause").show();
  }
  initAudio(next);
  audio.play();
  showDuration();
});

//Prev Button
$("#prev").click(function () {
  audio.pause();
  var prev = $("#playlist li.active").prev();
  if (prev.length == 0) {
    prev = $("#playlist li:last-child");
  }
  if ($("#play").is(":visible")) {
    $("#play").hide();
    $("#pause").show();
  }
  initAudio(prev);
  audio.play();
  showDuration();
});

//Playlist Song Click
$("#playlist li").click(function () {
  audio.pause();
  initAudio($(this));
  $("#play").hide();
  $("#pause").show();
  $("#duration").fadeIn(100);
  audio.play();
  showDuration();
});

//Volume Control
$("#volume").change(function () {
  audio.volume = parseFloat(this.value / 10);

  if (this.value == 0) {
    vol_btn.setAttribute("src", "/static/vol_off.png");
  } else {
    vol_btn.setAttribute("src", "/static/vol_on.png");
  }
});

//Time Duration
function showDuration() {
  $(audio).bind("timeupdate", function () {
    //Get minutes and seconds
    var s = parseInt(audio.currentTime % 60);
    var m = parseInt((audio.currentTime / 60) % 60);
    //Add 0 if seconds less than 10
    if (s < 10) {
      s = "0" + s;
    }
    $("#duration").html(m + ":" + s);
    var value = 0;
    if (audio.currentTime > 0) {
      value = (100 / audio.duration) * audio.currentTime;
    }
    $(".progress-bar").css("width", value + "%");
  });
}

$(".progress").mouseup(function (e) {
  var leftOffset = e.pageX - $(this).offset().left;
  var songPercents = leftOffset / $(".progress").width();
  try {
    audio.currentTime = songPercents * audio.duration;
  } catch (err) {
    console.log("Audio not loaded.");
  }
});

vol_btn.addEventListener("click", function (e) {
  e.preventDefault();
  if (is_on == 1) {
    vol_btn.setAttribute("src", "/static/vol_off.png");
    current_vol = document.getElementById("volume").value;
    document.getElementById("volume").value = 0;
    audio.volume = 0;
    is_on = 0;
  } else {
    vol_btn.setAttribute("src", "/static/vol_on.png");
    audio.volume = parseFloat(current_vol / 10);
    document.getElementById("volume").value = current_vol;
    is_on = 1;
  }
});
