import React from 'react'

import {Link} from 'react-router-dom'

import {List, Button, Header, Loader} from 'semantic-ui-react'
import 'semantic-ui-css/semantic.min.css'

import {auth, database} from '../../fire'

import {requests} from '../../requests'

import UserProfileImage from '../../components/UserProfileImage'

import EditableTextView from '../../components/EditableTextView'

import TwoChildrenPane from '../../components/TwoChildrenPane'

import {tsConfig} from '../../config'

import './UserProfile.css'

var profileStore = require('../../profileStore')

class UserProfile extends React.Component {
  constructor (props) {
    super(props)
    this.state = {
      user_id: props.userId,
      topics: [],
      user_name: profileStore.getName(props.user_Id),
      user_bio: '',
      loading_topics: true
    }
    this.fetchTopics = this.fetchTopics.bind(this)
    this.handleProfileTextSubmit = this.handleProfileTextSubmit.bind(this)

  }

  componentWillReceiveProps (nextProps) {
    if (this.state.user_id !== nextProps.userId) {
      const profileRef = database.ref(tsConfig.environment() + '/profiles/' + nextProps.userId)
      profileRef.once('value', (snapshot) => {
        this.setState({
          user_id: nextProps.userId,
          topics: [],
          user_name: snapshot.val().display_name,
          user_bio: snapshot.val().bio === '' ? 'Hey there! I\'m being Expertified!' : snapshot.val().bio,
          loading_topics: true
        })
      })
      this.fetchTopics(nextProps.userId)
    }
  }

  fetchTopics (userID) {
    let otherUser = userID
    let parent = this
    auth.currentUser.getIdToken(false).then(function (idToken) {
      parent.token = idToken
      requests.getUserChats(idToken, otherUser, (chats) => {
        parent.setState(prevState => ({token: idToken, topics: chats, loading_topics: false}))
      })
    }).catch(function (error) {
      // console.log(error)
    })
  }

  componentDidMount () {
    this.fetchTopics(this.state.user_id)
  }

  handleProfileTextSubmit (text) {
    requests.updateBio(this.state.token, text, function (res) {
    })
  }
  render () {
    let topicsDOM = <Loader active inline="centered"></Loader>
    let headerDOM =
        <TwoChildrenPane>
          <Button circular onClick={() => this.props.closeCallback()} icon="chevron left"/>
          <Header as='h2' style={{
            overflow: 'auto',
            width: 'auto',
            lineHeight: '66px'
          }} icon textAlign='center'>
            <Header.Content>
              {this.state.user_name}
            </Header.Content>
          </Header>
        </TwoChildrenPane>

    let userProfileTextDOM = null
    if (this.props.authUserId === this.state.user_id) {
      userProfileTextDOM =
            <EditableTextView
              onSubmit={this.handleProfileTextSubmit}
              text={this.state.user_bio}
            />
    } else {
      userProfileTextDOM =
            <Header as="h4">
              <Header.Content>
                {this.state.user_bio}
              </Header.Content>
            </Header>
    }

    let userActionDOM = null
    if (this.props.authUserId === this.state.user_id) {
      userActionDOM = <div></div>
    } else {
      userActionDOM =
        <div>
          <Button circular color="blue" onClick={() => this.props.handleDM(this.state.user_id)}>Message</Button>
          <Button circular color="red">Block</Button>
        </div>
    }
    if (!this.state.loading_topics) {
      topicsDOM = this.state.topics.map(topic => {
        return (<List.Item key={topic.id}>
          <List.Content>
            <List.Header as='a'><Link to={'/home/' + topic.id} key={topic.id}>{topic.subject}</Link></List.Header>
          </List.Content>
        </List.Item>)
      })
    }

    return (<div>
      {headerDOM}

      <div className="user-profile-image">
        <UserProfileImage user_id={this.state.user_id} size={200}/>
      </div>

      <div>
        <Header as="h5" style={{
          'padding': '0px 10px'
        }}>
          <Header.Content>
                  Profile text
          </Header.Content>
        </Header>

      </div>

      <div style={{'margin': '10px'}}>
        <span className="user-stat-value">
          {userProfileTextDOM}
        </span>
      </div>

      <div className="user-action">
        {userActionDOM}
      </div>

      <div className="user-stats">
        <div>
          <span className="user-stat">
            Topics created
          </span>
          <span className="user-stat-value">
            {this.state.topics.length}
          </span>
        </div>
      </div>
      <div className="topics-section">
        <TwoChildrenPane>
          <Button circular size="small" onClick={this.toggleUserProfilePane} icon="chevron down"/>
          <Header as='h4' style={{
            overflow: 'auto',
            width: 'auto',
            lineHeight: '66px'
          }} icon textAlign='center'>
            <Header.Content>
              Topics
            </Header.Content>
          </Header>
        </TwoChildrenPane>
        <List ordered style={{
          padding: '10px 10px'
        }}>
          {topicsDOM}
        </List>
      </div>
    </div>)
  }
}

export default UserProfile
