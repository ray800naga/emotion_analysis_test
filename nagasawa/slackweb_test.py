import slackweb
import getpass

slack = slackweb.Slack(url="https://hooks.slack.com/services/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")
slack.notify(text=getpass.getuser())