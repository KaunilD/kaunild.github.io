var TimeLine = function(){
  var experience = [
    {
      name:"iSenses Inc.",
      area:"Machine Learning and Computer Vision.",
      start:"Jan-15",
      end:"Jan-16",
      type:"Part-Time/Internship",
      side:"right",
      weight:1
    },
    {
      name:"Facebook Inc.",
      area:"Machine Learning and Computer Vision.",
      start:"Sep-16",
      end:"Sep-17",
      type:"Full-Time/Internship",
      side:"right",
      weight:2
    },
    {
      name:"Microsoft - Imagine Cup Korea Finalist, Team Cobrix",
      area:"Machine Learning and Computer Vision.",
      start:"May-17",
      end:"Jun-17",
      type:"Competition/Hackathon",
      side:"left",
      weight:20
    },
    {
      name:"B.E. Mumbai University",
      area:"Computer Engineering",
      start:"May-13",
      end:"Aug-17",
      type:"Education",
      side:"left",
      weight:0.5
    },
    {
      name:"Higher Secondary",
      area:"Computer Engineering",
      start:"Jan-11",
      end:"Feb-13",
      type:"Education",
      side:"left",
      weight:0.5
    }
  ]

  var width, height;
  var margin = {top:10, left:5, bottom:10, right:5};
  var svg;
  var scale = d3.scaleTime();
  var dateParser = d3.timeParse("%b-%y");
  var spline = d3.line().curve(d3.curveCardinal);

  var timeline = function(selector){
    width = selector.node().getBoundingClientRect().width - (margin.left + margin.right);
    height = selector.node().getBoundingClientRect().height - (margin.top + margin.bottom);
    scale.range([0, height]);
    svg = selector.append('svg')
      .attr('width', width)
      .attr('height', height)
      .append('g')
      .attr('transform', 'translate('+margin.left+ ', '+margin.top+')');

    experience.forEach(function(d){
      d.end = dateParser(d.end);
      d.start = dateParser(d.start);
    })

    scale.domain([dateParser("Jan-11"), dateParser("Dec-17")]);
    svg.append('path')
      .datum([[width/2, 0], [width/2, height]])
      .attr('d', d3.line())
      .attr('id', 'timeline-axis');



    var spline_container = svg.append('g')
      .attr('id','spline-container');


    spline_container.selectAll('path#workex-spline')
      .data(experience)
      .enter()
      .append('path')
      .attr('id', 'workex-spline')
      .attr('d', function(d, i){

        if (d.side === "right"){
          return spline([
            [width/2, scale(d.start)],
            [width/2 + d3.min([(-scale(d.start)+scale(d.end))*d.weight, width/2]), (scale(d.start)+ scale(d.end))/2],
            [width/2, scale(d.end)]
          ])
        }else{
          return spline([
            [width/2, scale(d.start)],
            [width/2 - d3.min([(-scale(d.start)+scale(d.end))*d.weight, width/2]), (scale(d.start)+ scale(d.end))/2],
            [width/2, scale(d.end)]
          ])
        }
      })

      var ticks = [];

      for (var i = 0; i <= scale(scale.domain()[1]) - scale(scale.domain()[0]); i+=(scale(scale.domain()[1]) - scale(scale.domain()[0]))/8){
        ticks.push(i);
      }

      var ticks_container = svg.append('g')
        .attr('id', 'tick-container');

      ticks_container.selectAll('circle#filled')
        .data(ticks)
        .enter()
        .append('circle')
        .attr('id', 'filled')
        .attr('cx', width/2)
        .attr('cy', function(d, i){return d;})
        .attr('r', '2');

      ticks_container.selectAll('circle#stroked')
        .data(ticks)
        .enter()
        .append('circle')
        .attr('id', 'stroked')
        .attr('cx', width/2)
        .attr('cy', function(d, i){return d;})
        .attr('r', '5');


  }
  return timeline;
}
