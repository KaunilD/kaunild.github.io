var Sidebar = function(){
  var sidebar = function(selector){
    var container = selector.append('div').attr('id','sidebar');

    var profile = container.append('div').attr('id', 'profile');
    var profile_data = ["Kaunil D.", "9987310565", "dhruv.kaunil@gmail.com", "Mumbai. Maha. India."];
    profile.append('img').attr('id', 'profile-image');
    var profile_dets_list = profile.append('ul').attr('id', 'profile-dets');
    profile_dets_list.selectAll('li')
      .data(profile_data)
      .enter()
      .append('li')
      .attr('id', 'profile-dets-item')
      .text(function(d){return d;});

    var navigation = container.append('ul').attr('id', 'sidebar-nav');
    var nav_data = ["Experience", "Projects", "Publications","Education"];

    navigation.selectAll('li')
      .data(nav_data)
      .enter()
      .append('li')
      .attr('id', 'sidebar-nav-item')
      .text(function(d){return d;})
  }
  return sidebar;
}
