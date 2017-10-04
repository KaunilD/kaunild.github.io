var Ui = function(){

  var ui = function(selector){

    selector.call(Sidebar());
    selector.call(Content);

  }
  return ui;

}
